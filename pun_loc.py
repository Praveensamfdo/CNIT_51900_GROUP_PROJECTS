import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import xml.etree.ElementTree as ET
from transformers import BertModel
from transformers import BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 1000

############################################################################################################
# Data preparation

path1_test = 'datasets/subtask1-heterographic-test.xml'
path1_gold = 'datasets/subtask1-heterographic-test.gold'
path2_test = 'datasets/subtask2-heterographic-test.xml'
path2_gold =  'datasets/subtask2-heterographic-test.gold'

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

############################################################################################################
# NN models

class BERT(pl.LightningModule):
    def __init__(self, output_size):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions = True)
        config = self.bert.config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.linear2 = nn.Linear(config.hidden_size // 2, 3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bert(**x)
        x = self.linear(x[0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        sents = batch['sentence'].to(device)
        locations = batch['location'].to(device)
        output = self(sents)

        gtruth = torch.zeros(output.shape[1]).to(device).long()
        gtruth[locations[0]] = 1
        gtruth[locations[0] + 1:] = 2

        loss = self.ce_criterion(output.reshape(output.shape[1], output.shape[2]), gtruth)
        _, preds = torch.max(output.data, 2)
        self.train_acc(preds, gtruth.reshape(preds.shape).long())
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sents = batch['sentence'].to(device)
        locations = batch['location'].to(device)
        output = self(sents)

        gtruth = torch.zeros(output.shape[1]).to(device).long()
        gtruth[locations[0]] = 1
        gtruth[locations[0] + 1:] = 2

        loss = self.ce_criterion(output.reshape(output.shape[1], output.shape[2]), gtruth)
        _, preds = torch.max(output.data, 2)
        self.val_acc(preds, gtruth.reshape(preds.shape).long())
        self.log('test_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

early_stopping = EarlyStopping(monitor = 'test_acc', mode = 'max', stopping_threshold = 0.95, patience = 1e9) # Set patience to large number to force early stopping only on stopping_threshold

checkpoint_callback = ModelCheckpoint(
    monitor="test_acc",
    dirpath="models/",
    filename="model-{epoch:03d}-{test_acc:.4f}",
    save_top_k=1,
    mode="max",
)

############################################################################################################

train_data_loc = PunDataset(train_data)
test_data_loc = PunDataset(test_data)
trainloader = torch.utils.data.DataLoader(train_data_loc, batch_size=1, shuffle=True, num_workers=4, collate_fn = collate_fn)
testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_fn)

model = LightningModel()
trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS, callbacks=[checkpoint_callback, early_stopping], log_every_n_steps=10)
trainer.fit(model, trainloader, testloader)