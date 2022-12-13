"""
***Flask server for pun retrieval***
"""
# Import required libraries
import re
import csv
import yake
import nltk
import tqdm
import torch
import spacy
import numpy as np
import torchmetrics
import torch.nn as nn
from flask import Flask
from flask import request
from abydos import phonetic
import pytorch_lightning as pl
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, BertModel

app = Flask(__name__)
custom_kw_extractor = yake.KeywordExtractor(lan='en', n = 1, dedupLim = 0.9, top = 20, features=None)
stemmer = nltk.stem.snowball.SnowballStemmer("english")
model = SentenceTransformer('bert-base-nli-mean-tokens')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")

PUN_SENT = None

PROCESSED_DATA = 'processed_data'
BERT_MOD = 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(BERT_MOD)
mlm_bert = BertForMaskedLM.from_pretrained(BERT_MOD).to(device)
mlm_bert.eval()

##################################################################
######################## Helper functions ########################
##################################################################

##################################################################
# Source detection neural network model

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

net = LightningModel().load_from_checkpoint('models/model-epoch=004-test_acc=0.9160.ckpt')

##################################################################   
# LESK word sense disambiguation

stopwords_en = set(stopwords.words('english'))

def tokenize(document: str, word: str) -> set:
    # obtaining tokens from the gloss
    tokenizer_nltk = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer_nltk.tokenize(document)

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

def kword_extract(corpus):
    kwstem_sent_map = {}
    kwstem_kw_map = {}

    for item in tqdm.tqdm(corpus):
        sent = " ".join(item['sentence'])
        keywords = custom_kw_extractor.extract_keywords(sent)

        for kw in keywords:
            kw_stem = stemmer.stem(kw[0])
            if kw_stem in kwstem_sent_map:
                kwstem_sent_map[kw_stem].append([item['sentence'], item['src']])

            else:
                kwstem_sent_map[kw_stem] = [[item['sentence'], item['src']]]
                kwstem_kw_map[kw_stem] = kw[0]

    return kwstem_sent_map, kwstem_kw_map

def get_sents(kword_sent_map, kwstem_kw_map, keys, key_embeddings, category):
    category = stemmer.stem(category)
    cat_embedding = model.encode([category])
    c_sim = cosine_similarity(cat_embedding, key_embeddings)

    # Indices of the top cosine similarity values
    maxsim_idx = np.argmax(c_sim[0])
    max_kword = keys[maxsim_idx]
    max_score = c_sim[0][maxsim_idx]
    rand_sent = kword_sent_map[max_kword][np.random.randint(0, len(kword_sent_map[max_kword]))]
    print("Max keyword: %s | max score: %s" %(max_kword, max_score))
    return [" ".join(rand_sent[0]), str(max_score), kwstem_kw_map[max_kword], rand_sent[1]]

def data_prep():
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
        all_data.append({"sentence": sent_list, "src": sent_list[src_tgt_dict[key][2] - 1], "src_root": src_tgt_dict[key][0], "tgt_root": src_tgt_dict[key][1]})

    return all_data

pun_corpus = data_prep()
kword_sent_map, kwstem_kw_map = kword_extract(pun_corpus)
keys = list(kword_sent_map.keys())
key_embeddings = model.encode(keys)
print("Total keywords: %s" %len(kword_sent_map))

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
        self.ss_algos = [phonetic.NYSIIS(), phonetic.Soundex(), phonetic.Caverphone(), phonetic.Metaphone()]
        self.ss_weights = [0.1, 0.2, 0.2, 0.5]
        self.word_dict = {}
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")    # Stemmer to find the root of a word

        with open('all_words.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.word_dict[row[0]] = {'word_root': self.stemmer.stem(row[0]), 'encodes': [self.ss_algos[0].encode(row[0]), self.ss_algos[1].encode(row[0]), 
                                                            self.ss_algos[2].encode(row[0]), self.ss_algos[3].encode(row[0])]}

    def sim_words(self, src_word):
        sim_word_list = []
        src_root = self.stemmer.stem(src_word)
        src_encodes = [self.ss_algos[0].encode(src_word), self.ss_algos[1].encode(src_word), self.ss_algos[2].encode(src_word), self.ss_algos[3].encode(src_word)]
        
        for word in self.word_dict.keys():
            word_root = self.word_dict[word]['word_root']
            if word_root != src_root:
                word_encodes = self.word_dict[word]['encodes']

                tot_weight = 0

                for src_encode, word_encode, weight in zip(src_encodes, word_encodes, self.ss_weights):
                    lev_dist = edit_distance (src_encode, word_encode)
                    tot_weight += lev_dist * weight

                if tot_weight <= self.max_diff:
                    sim_word_list.append(word)

        return sim_word_list

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
    inputs = tokenizer(masked_sent, return_tensors="pt").to(device)
    loss_arr = []

    for cand in tqdm.tqdm(cand_list):
        sent_option = masked_sent.replace("[MASK]", cand)
        labels = tokenizer(sent_option, return_tensors="pt")["input_ids"][0][:inputs.input_ids.shape[1]].to(device)

        try:
            outputs = mlm_bert(**inputs, labels=labels)
            loss = outputs.loss.item()

        except:
            loss = 1000

        loss_arr.append(loss)

    min_idx = torch.argmin(torch.tensor(loss_arr)).item()
    pred_word = cand_list[min_idx]
    pred_sent = masked_sent.replace('[MASK]', pred_word)
    pred_word = re.sub(r'[^\w\s]', '', pred_word)           # Replace any punctuations with ''

    return pred_word, pred_sent

word_sim = GetSimWords(1.0)

##################################################################
######################## Flask functions #########################
##################################################################

@app.route("/getapun")
def get_pun_sent():
    data = request.args
    puncat = data['puncat']
    pun_item = get_sents(kword_sent_map, kwstem_kw_map, keys, key_embeddings, puncat)
    global PUN_SENT
    PUN_SENT = pun_item[0]
    return pun_item
 
@app.route("/getpunexp")
def exp_pun_sent():
    print("Pun sentence: ", PUN_SENT)
    """
    Sound similarity-based src-tgt identification
    """
    src_sent = PUN_SENT

    ##########################################
    ########## Source pun detection ##########
    ##########################################

    sent_tok_out = tokenizer(src_sent, return_tensors="pt", padding = True).to(device)
    output = net(sent_tok_out)
    _, preds = torch.max(output.data, 2)
    preds = preds.cpu().numpy().tolist()[0]

    print("Source prediction array: ", preds)

    pred_loc = preds.index(1)
        
    if pred_loc + 1 == len(src_sent.split()):
        pred_loc -= 1

    doc = nlp(src_sent)
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
            try:
                if left_right.index('VERB') == 0:
                    pred_loc -= 1
                else:
                    pred_loc += 1

            except:
                pass

    src = src_sent.split()[pred_loc]
    print("Predicted source word: ", src)

    ##########################################
    ###### Target pun detection and WSD ######
    ##########################################

    masked_sent = src_sent.replace(src, '[MASK]')

    cand_list = word_sim.sim_words(src)
    print("Candidate list: ", cand_list)

    if cand_list != []:
        tgt, tgt_sent = get_target_pun(masked_sent, cand_list)
        
        print("Source word: ", src)
        print("Target word: ", tgt)
        print("Target sentence: ", tgt_sent)

        try:
            src_sense, _ = simple_lesk(src_sent, src)
            src_sense_def = src_sense.definition()

        except:
            src_sense_def = None

        try:
            tgt_sense, _ = simple_lesk(tgt_sent, tgt)
            tgt_sense_def = tgt_sense.definition()

        except:
            tgt_sense_def = None

        
        if src_sense_def == None and tgt_sense_def != None:
            return "The source is " + src + ". The target is " + tgt + " with the gloss:   " + tgt_sense_def + ". Do you want to exit?"

        elif src_sense_def != None and tgt_sense_def == None:
            return "The source is " + src + " with the gloss:   " + src_sense_def + ". The target is " + tgt + ". Do you want to exit?"

        elif src_sense_def == None and tgt_sense_def == None:
            return "The source is " + src + ". The target is " + tgt + ". Do you want to exit?"

        else:
            return "The source is " + src + " with the gloss:   " + src_sense_def + ". The target is " + tgt + " with the gloss:   " + tgt_sense_def + ". Do you want to exit?"

    else:
        try:
            src_sense, _ = simple_lesk(src_sent, src)
            src_sense_def = src_sense.definition()

        except:
            src_sense_def = None

        if src_sense_def == None:
            return "The source is " + src + ". Do you want to exit?"

        else:
            return "The source is " + src + " with the gloss:   " + src_sense_def + ". Do you want to exit?"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7500, debug=True)