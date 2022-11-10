import os
import re
import csv
import time
import nltk
import torch
import random
import numpy as np
from tqdm import tqdm
from nltk.wsd import lesk
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForMaskedLM

nltk.download('all')
nltk.download('cmudict')

############################################################################################################
# Constants and hyperparameters

PROCESSED_DATA = 'processed_data'
BERT_MOD = 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(BERT_MOD)
model = BertForMaskedLM.from_pretrained(BERT_MOD)
model.eval()

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
    path3_test = 'datasets/subtask3-heterographic-test.xml'
    path3_gold = 'datasets/subtask3-heterographic-test.gold'

    sent_dict = {}
    src_tgt_dict = {}

    # Get the pun sentences

    tree3 = ET.parse(path3_test)
    root3 = tree3.getroot()

    for child in root3:
        idx = child.attrib["id"]
        line = []
        for kid in child:
            line.append(kid.text)

        sent_dict[idx] = line

    # Get the source and target puns

    with open(path3_gold) as gold3:
        lines = gold3.readlines()
        for line in lines:
            token = line.strip().split("\t")
            _, idx, loc = token[0].split("_")
            src = token[1].split('%')[0]
            tgt = token[2].split('%')[0]
            src_tgt_dict['het_' + str(idx)] = (src, tgt, int(loc))

    all_data = []

    for key, sent_list in sent_dict.items():
        all_data.append({"sentence": " ".join(sent_list), "src": sent_list[src_tgt_dict[key][2] - 1], "src_root": src_tgt_dict[key][0], "tgt_root": src_tgt_dict[key][1]})

    return all_data

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
        sample = {'sentence' : sample_item['sentence'], 'src' : sample_item['src'], 'src_root' : sample_item['src_root'], 'tgt_root' : sample_item['tgt_root']}
        return(sample)

############################################################################################################
# Word similarity methods (inspired by: https://github.com/maxwell-schwartz/PUNchlineGenerator)

def edit_distance(w1, w2):
    cost = np.zeros((len(w1) + 1, len(w2) + 1), dtype = np.int32)
    
    for i in range(1, len(w1) + 1):
        cost[i][0] = i

    for j in range(1, len(w2) + 1):
        cost[0][j] = j

    cost = cost.tolist()

    # Baseline costs
    del_cost = 1
    add_cost = 1
    sub_cost = 2
    
    for i in range(1, len(w1) + 1):
        for j in range(1, len(w2) + 1):
            if w1[i-1] == w2[j-1]:
                sub_cost = 0
            else:
                sub_cost = 2

            # Get the totals
            del_total = cost[i-1][j] + del_cost
            add_total = cost[i][j-1] + add_cost
            sub_total = cost[i-1][j-1] + sub_cost

            # Choose the lowest cost from the options
            options = [del_total, add_total, sub_total]
            options.sort()
            cost[i][j] = options[0]

    return cost[-1][-1]

class GetSimWords():
    def __init__(self, max_diff):
        self.max_diff = max_diff
        prondict = nltk.corpus.cmudict.dict()                           # CMU Pronunciation Dictionary
        word_dict = {}                                                  # Dictionary taken from PunchlineGenerator

        with open('all_words.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                word_dict[row[0]] = [eval(row[1])]

        self.final_prondict = prondict | word_dict                      # Final pronounciation dictionary
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")    # Stemmer to find the root of a word

    def sim_words(self, src_word):
        sim_word_list = []

        try:
            src_pron = self.final_prondict[src_word][0] 
            src_root = self.stemmer.stem(src_word)
         
            for word in self.final_prondict.keys():
                word_root = self.stemmer.stem(word)
                if word_root != src_root:
                    word_pron = self.final_prondict[word][0] 
                    price = edit_distance(src_pron, word_pron)
                    if price <= self.max_diff:
                        sim_word_list.append(word)

        except:
            pass

        return sim_word_list

word_sim = GetSimWords(2)

############################################################################################################    
# LESK word sense disambiguation

stopwords_en = set(stopwords.words('english'))

def tokenize(document: str, word: str) -> set:
    # obtaining tokens from the gloss
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(document)

    # removing stop words from tokens
    tokens = [token for token in tokens if token not in stopwords_en and token.isalpha()]

    # removing the word from the tokens
    tokens = [token for token in tokens if token != word]
    return set(tokens)

def simple_lesk(gloss: str, word: str):
    """":returns the sense most suited to the given word as per the Simple LESK Algorithm"""

    # converting everything to lowercase
    gloss = gloss.lower()
    word = word.lower()

    # obtaining tokens from the gloss
    gloss_tokens = tokenize(gloss, word)

    # calculating the word sense disambiguation using simple LESK
    synsets = wordnet.synsets(word)
    weights = [0] * len(synsets)
    N_t = len(synsets)
    N_w = {}

    # Creating the IDF Frequency column using Laplacian Scaling
    for gloss_token in gloss_tokens:
        N_w[gloss_token] = 1

        for sense in synsets:
            if gloss_token in sense.definition():
                N_w[gloss_token] += N_t
                continue

            for example in sense.examples():
                if gloss_token in example:
                    N_w[gloss_token] += N_t
                    break

    for index, sense in enumerate(synsets):
        # adding tokens from examples into the comparison set
        comparison = set()
        for example in sense.examples():
            for token in tokenize(example, word):
                comparison.add(token)

        # adding tokens from definition into the comparison set
        for token in tokenize(sense.definition(), word):
            comparison.add(token)

        # comparing the gloss tokens with comparison set
        for token in gloss_tokens:
            if token in comparison:
                weights[index] += np.log(N_w[token] / N_t)

    max_weight = max(weights)
    index = weights.index(max_weight)
    return synsets[index], weights

############################################################################################################
# Algorithm 1: target pun word prediction

def get_target_pun(masked_sent, cand_list):
    """
    *** Algorithm 1 ***
    * Inputs:
        - masked_sent: the sentence with pun word masked
        - cand_list: list of candidate pun words

    * Outputs:
        - pred_word: the predicted target pun word
        - pred_sent: the predicted sentence with the target pun word
    """
    inputs = tokenizer(masked_sent, return_tensors="pt")
    loss_arr = []

    for cand in tqdm(cand_list):
        sent_option = masked_sent.replace("[MASK]", cand)
        labels = tokenizer(sent_option, return_tensors="pt")["input_ids"][0][:inputs.input_ids.shape[1]]
        outputs = model(**inputs, labels=labels)
        loss_arr.append(outputs.loss.item())

    min_idx = torch.argmin(torch.tensor(loss_arr)).item()
    pred_word = cand_list[min_idx]
    pred_sent = masked_sent.replace('[MASK]', pred_word)
    pred_word = re.sub(r'[^\w\s]', '', pred_word)           # Replace any punctuations with ''

    return pred_word, pred_sent
    
############################################################################################################

all_data = data_prep(test_percent = 0.2, processed_data = PROCESSED_DATA)
train_data_loc = PunDataset(all_data)
stemmer = nltk.stem.snowball.SnowballStemmer("english")
cumulative_acc = 0

for idx, train in enumerate(train_data_loc):
    sent = train['sentence']
    src = train['src']
    tgt_root = train['tgt_root']
    masked_sent = sent.replace(src, '[MASK]')
    cand_list = word_sim.sim_words(src)

    if cand_list != []:
        target_pred_pun, target_sent = get_target_pun(masked_sent, cand_list)
        print("Sentence: ", sent)
        print("Source word: ", src)
        print("Target root: ", tgt_root)
        print("\nPredicted target: ", target_pred_pun)

        try:
            sense, weights = simple_lesk(sent, src)
            print("Source sense: ", sense.definition())
        
        except:
            print("Source sense: ", None)

        try:
            sense, weights = simple_lesk(target_sent, target_pred_pun)
            print("Target sense: ", sense.definition())

        except:
            print("Target sense: ", None)

        tgt_stem = stemmer.stem(tgt_root)
        pred_stem = stemmer.stem(target_pred_pun)

        if tgt_stem == pred_stem:
            cumulative_acc += 1

        print("\nNumber of sentences checked: %s | stemps: (%s, %s) | accuracy percentage: %4.2f" %(idx + 1, tgt_stem, pred_stem, (cumulative_acc/(idx + 1)) * 100))

        print("===========================================================")