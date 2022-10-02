import os
import torch
import pickle
import random
import numpy as np
import torchmetrics
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from transformers import BertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
EPOCHS = 1000

############################################################################################################
# Set the seed for reproducibility

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################################################
# Data preparation methods

def data_prep():
    if os.path.exists("train_data_det.pkl") and os.path.exists("test_data_det.pkl"):
        with open("train_data_det.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open("test_data_det.pkl", "rb") as f:
            test_data = pickle.load(f)
    
    else:
        path1_test = 'datasets/subtask1-homographic-test.xml'
        path1_gold = 'datasets/subtask1-homographic-test.gold'

        all_instances = {}
        classes = {}

        tree1 = ET.parse(path1_test)
        root1 = tree1.getroot()

        for child in root1:
            idx = child.attrib["id"]
            line = []
            for kid in child:
                line.append(kid.text)
            all_instances[idx] = line

        with open(path1_gold) as gold1:
            lines = gold1.readlines()
            for line in lines:
                token = line.strip().split("\t")
                classes[token[0]] = token[1]

        all_data = []

        for idx in all_instances.keys():
            sentence = " ".join(all_instances[idx])
            label = int(classes[idx])
            all_data.append({"sentence": sentence, "label": label})

        # Randomize the data
        random.shuffle(all_data)
        percent = 0.2
        train_data = all_data[:int(len(all_data) * percent)]
        test_data = all_data[int(len(all_data) * percent):]

        # Save train and test data as pkls
        with open("train_data_det.pkl", "wb") as f:
            pickle.dump(train_data, f)

        with open("test_data_det.pkl", "wb") as f:
            pickle.dump(test_data, f)

    return train_data, test_data
    
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
        sample = {'sentence' : sentence, 'label' : label}
        return(sample)

def collate_fn(batch):
    labels = []
    sentences = []

    for item in batch:
        labels.append(item['label'])
        sentences.append(item['sentence'])

    # Stack the labels and tokenize the sentences
    labels = torch.stack(labels, 0)
    sent_tok_out = tokenizer(sentences, return_tensors="pt", padding = True)
    return {'label': labels, 'sentence': sent_tok_out}

############################################################################################################
# NN models

class BERT(pl.LightningModule):
    def __init__(self, output_size):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions = True)
        config = self.bert.config
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.linear2 = nn.Linear(config.hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bert(**x)[1]
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.nlp_trans = BERT(output_size = 2).to(device)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, sents):
        classi_out = self.nlp_trans(sents)
        return(classi_out)

    def training_step(self, batch, batch_idx):
        labels = batch['label'].to(device)
        sents = batch['sentence'].to(device)
        output = self(sents)

        loss = self.ce_criterion(output, labels)

        _, preds = torch.max(output.data, 1)
        self.train_acc(preds, labels.reshape(preds.shape).long())
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label'].to(device)
        sents = batch['sentence'].to(device)
        output = self(sents)

        loss = self.ce_criterion(output, labels)

        _, preds = torch.max(output.data, 1)
        self.val_acc(preds, labels.reshape(preds.shape).long())
        self.log('test_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

############################################################################################################

_, test_data = data_prep()
test_data_loc = PunDataset(test_data)
testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=1, shuffle=False, num_workers=4, collate_fn = collate_fn)

net = LightningModel().load_from_checkpoint('models_det/model-epoch=029-test_acc=0.9033.ckpt')
net.eval()

all_corr = 0
TP = TN = FP = FN = 0

for batch in tqdm(testloader):
    sents = batch['sentence'].to(device)
    labels = batch['label'].to(device)
    output = net(sents)
    _, preds = torch.max(output.data, 1)
    
    if preds == labels:
        all_corr += 1

    if preds == 1 and labels == 1:
        TP += 1
    elif preds == 0 and labels == 0:
        TN += 1
    elif preds == 1 and labels == 0:
        FP += 1
    elif preds == 0 and labels == 1:
        FN += 1
 
print("Testing accuracy: ", (all_corr / len(testloader)) * 100)

# Print confusion matrix in Matplotlib
cm = np.array([[TP, FP], [FN, TN]])

# Normalize the confusion matrix
cm = (cm.astype('float') / cm.sum(axis=0)) * 100

df_cm = pd.DataFrame(cm, index = [i for i in ["Pun", "Not Pun"]],
                    columns = [i for i in ["Pun", "Not Pun"]])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='g')

# Add x-axis and y-axis labels
plt.ylabel('Predicted labels')
plt.xlabel('True labels')
plt.show()