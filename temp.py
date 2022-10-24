import os
import torch
import pickle
import random
import numpy as np
import xml.etree.ElementTree as ET

PROCESSED_DATA = 'processed_data'

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

def data_prep(test_percent, processed_data):
    if os.path.exists(processed_data + "/train_data_hetero.pkl") and os.path.exists(processed_data + "/test_data_hetero.pkl"):
        with open(processed_data + "/train_data_hetero.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(processed_data + "/test_data_hetero.pkl", "rb") as f:
            test_data = pickle.load(f)
    
    else:
        path1_test = 'datasets/subtask1-heterographic-test.xml'
        path1_gold = 'datasets/subtask1-heterographic-test.gold'
        path2_test = 'datasets/subtask2-heterographic-test.xml'
        path2_gold =  'datasets/subtask2-heterographic-test.gold'

        all_instances = {}
        pun_instances = {}
        classes = {}
        locations = {}

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

        tree2 = ET.parse(path2_test)
        root2 = tree2.getroot()

        for child in root2:
            idx = child.attrib["id"]
            line = []
            for kid in child:
                line.append(kid.text)
            pun_instances[idx] = line

        with open(path2_gold) as gold2:
            lines = gold2.readlines()
            for line in lines:
                token = line.strip().split("\t")
                sub_tokens = token[1].split("_")
                locations[token[0]] = sub_tokens[2]

        all_data = []

        for idx in all_instances.keys():
            sentence = " ".join(all_instances[idx])
            label = int(classes[idx])
            pun_word = pun_instances[idx][int(locations[idx]) - 1] if label == 1 else None      # If the sentence is a pun, get the pun word, otherwise set it to None
            pun_location = int(locations[idx]) - 1 if label == 1 else None                      # If there is no pun, set the location to None (pun location is 0-indexed)
            all_data.append({"sentence": sentence, "label": label, "pun_word": pun_word, "pun_location": pun_location})

        # Randomize the data
        random.shuffle(all_data)
        percent = test_percent
        train_data = all_data[:int(len(all_data) * percent)]
        test_data = all_data[int(len(all_data) * percent):]

        # Save train and test data as pkls
        with open(processed_data + "/train_data_hetero.pkl", "wb") as f:
            pickle.dump(train_data, f)

        with open(processed_data + "/test_data_hetero.pkl", "wb") as f:
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
        sample = {'sentence' : sample_item['sentence'], 'label' : label, 'pun_word' : sample_item['pun_word'], 'pun_location' : sample_item['pun_location']}
        return(sample)

############################################################################################################

train_data, test_data = data_prep(test_percent = 0.2, processed_data = PROCESSED_DATA)
train_data_loc = PunDataset(train_data)
test_data_loc = PunDataset(test_data)

for idx, train in enumerate(train_data_loc):
    print(train)
    if idx == 10:
        break