import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import BertModel
from transformers import BertTokenizer
import xml.etree.ElementTree as ET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
############################################################################################################
# Data preparation

path1_test = 'datasets/subtask1-homographic-test.xml'
path1_gold = 'datasets/subtask1-homographic-test.gold'
path2_test = 'datasets/subtask2-homographic-test.xml'
path2_gold =  'datasets/subtask2-homographic-test.gold'

all_instances = {}
pun_instances = {}
classes = {}
locations = {}
max_sent_len = None

tree1 = ET.parse(path1_test)
root1 = tree1.getroot()

for child in root1:
    idx = child.attrib["id"]
    line = []
    for kid in child:
        line.append(kid.text)

    max_sent_len = len(line) if max_sent_len is None else max(len(line), max_sent_len)
    all_instances[idx] = line

tree2 = ET.parse(path2_test)
root2 = tree2.getroot()

for child in root2:
    line = []
    idx = child.attrib["id"]
    for kid in child:
        line.append(kid.text)

    pun_instances[idx] = line

with open(path1_gold) as gold1:
    lines = gold1.readlines()
    for line in lines:
        token = line.strip().split("\t")
        classes[token[0]] = token[1]

with open(path2_gold) as gold2:
    lines = gold2.readlines()
    for line in lines:
        token = line.strip().split("\t")
        sub_tokens = token[1].split("_")
        locations[token[0]] = sub_tokens[2]

all_data = []

for idx in pun_instances.keys():
    sentence = " ".join(pun_instances[idx])
    label = int(classes[idx])
    pun_word = pun_instances[idx][int(locations[idx]) - 1] if label == 1 else None      # If the sentence is a pun, get the pun word, otherwise set it to None
    pun_location = int(locations[idx]) - 1 if label == 1 else 0                         # If there is no pun, set the location to 0
    all_data.append({"sentence": sentence, "label": label, "location": pun_location})

print('Maximum sentence length: %s' % max_sent_len)

# Randomize the data
#random.shuffle(all_data)
percent = 0.2
train_data = all_data[:int(len(all_data) * percent)]
test_data = all_data[int(len(all_data) * percent):]

class PunDataset(torch.utils.data.Dataset):
    """
    data loader class
    """
    def __init__(self, data_list):
        self.dataset = data_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_item = self.dataset[idx]
        label = sample_item['label']
        label = torch.tensor(label, dtype=torch.long)
        sentence = sample_item['sentence']
        location = sample_item['location']
        location = torch.tensor(location, dtype=torch.long)
        sample = {'sentence' : sentence, 'label' : label, 'location' : location}
        return(sample)

def collate_fn(batch):
    labels = []
    sentences = []
    locations = []

    for item in batch:
        labels.append(item['label'])
        sentences.append(item['sentence'])
        locations.append(item['location'])

    # Stack the labels and tokenize the sentences
    labels = torch.stack(labels, 0)
    locations = torch.stack(locations, 0)
    sent_tok_out = tokenizer(sentences, return_tensors="pt", padding = True)
    return {'label': labels, 'sentence': sent_tok_out, 'location': locations}

test_data_loc = PunDataset(test_data)
testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=1, shuffle=False, num_workers=4)

class BERT(pl.LightningModule):
    def __init__(self, output_size):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions = True)
        config = self.bert.config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.linear2 = nn.Linear(config.hidden_size // 2, 3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bert(**x)
        x = self.linear(x[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        #x = self.sigmoid(x)
        return x

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.nlp_trans = BERT(output_size = 69).to(device)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, sents):
        classi_out = self.nlp_trans(sents)
        return(classi_out)

    def training_step(self, batch, batch_idx):
        #labels = batch['label'].to(device)
        sents = batch['sentence'].to(device)
        locations = batch['location'].to(device)
        output = self(sents)

        gtruth = torch.zeros(output.shape[1]).to(device).long()
        gtruth[locations[0]] = 1
        gtruth[locations[0] + 1:] = 2

        loss = self.ce_criterion(output.reshape(output.shape[1], output.shape[2]), gtruth)
        #loss = self.bce_criterion(output, gtruth)

        _, preds = torch.max(output.data, 2)
        #preds = (output > 0.5).long()
        self.train_acc(preds, gtruth.reshape(preds.shape).long())
        #self.train_acc(preds, locations.reshape(preds.shape).long())
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #labels = batch['label'].to(device)
        sents = batch['sentence'].to(device)
        locations = batch['location'].to(device)
        output = self(sents)

        gtruth = torch.zeros(output.shape[1]).to(device).long()
        gtruth[locations[0]] = 1
        gtruth[locations[0] + 1:] = 2

        loss = self.ce_criterion(output.reshape(output.shape[1], output.shape[2]), gtruth)
        #loss = self.bce_criterion(output, gtruth)

        _, preds = torch.max(output.data, 2)

        #preds = (output > 0.5).long()
        self.val_acc(preds, gtruth.reshape(preds.shape).long())

        #self.val_acc(preds, locations.reshape(preds.shape).long())
        self.log('test_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

net = LightningModel().load_from_checkpoint('models/model-epoch=003-test_acc=0.8973.ckpt')
#trainer = pl.Trainer(gpus=1)
#trainer.validate(net, dataloaders=testloader)
all_corr = 0
all_inc = 0

import time
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

print(len(test_data_loc))

for idx, data in enumerate(testloader):
    sents = data['sentence']
    locations = data['location']
    sent_tok_out = tokenizer(sents, return_tensors="pt", padding = True).to(device)
    output = net(sent_tok_out)
    _, preds = torch.max(output.data, 2)

    preds = preds.cpu().numpy().tolist()[0]

    try:
        pred_loc = preds.index(1)
        
        if pred_loc + 1 == len(sents[0].split()):
            pred_loc -= 1

        doc = nlp(sents[0])
        pos = [token.pos_ for token in doc]
        lemma = [token.lemma_ for token in doc]

        if lemma[pred_loc] == 'the' or lemma[pred_loc] == "'" or lemma[pred_loc] == 'to' or lemma[pred_loc] == 'a' or lemma[pred_loc] == 'of' or pos[pred_loc] == 'PRON':
            left_right = [pos[pred_loc  - 1], pos[pred_loc + 1]]

            if 'NOUN' in left_right:
                if left_right.index('NOUN') == 0:
                    pred_loc -= 1
                else:
                    pred_loc += 1
            
            else:
                if left_right.index('VERB') == 0:
                    pred_loc -= 1
                else:
                    pred_loc += 1

        gtruth = locations.cpu().numpy().tolist()[0]
        print("=============================================")
        print("Sentence:\t\t%s" %sents[0])
        print("Predicted pun word:\t%s" %sents[0].split()[pred_loc])
        print("Ground truth:\t\t%s" %sents[0].split()[gtruth])
        print(gtruth)
        print(pred_loc)
        time.sleep(1)

    except:
        pass