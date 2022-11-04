from transformers import BertTokenizer, BertForMaskedLM
import torch
import csv
from gingerit.gingerit import GingerIt

"""
***Code from puchline generator.
"""

def edit_distance(w1, w2):
    
    cost = []
    
    # These may be useful for later work:
    #vowels = ['A', 'E', 'I', 'O', 'U']
    #voiced = ['B', 'D', 'G', 'J', 'L', 'M', 'N', 'R', 'V', 'W', 'Y', 'Z']
    #unvoiced = ['C', 'F', 'H', 'K', 'P', 'S', 'T']
    
    for i in range(len(w1)+1):
        x = []
        for j in range(len(w2)+1):
            x.append(0)
        cost.append(x)
    
    for i in range(len(w1)+1):
        cost[i][0] = i
    for j in range(len(w2)+1):
        cost[0][j] = j
        
    # baseline costs
    del_cost = 1
    add_cost = 1
    sub_cost = 2
    
    for i in range(1, len(w1)+1):
        for j in range(1, len(w2)+1):
            if w1[i-1] == w2[j-1]:
                sub_cost = 0
            else:
                sub_cost = 2
            # get the totals
            del_total = cost[i-1][j] + del_cost
            add_total = cost[i][j-1] + add_cost
            sub_total = cost[i-1][j-1] + sub_cost
            # choose the lowest cost from the options
            options = [del_total, add_total, sub_total]
            options.sort()
            cost[i][j] = options[0]

    return cost[-1][-1]

real_list = []
real_words = []
with open('all_words.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        real_list.append((row[0], eval(row[1]), row[2]))
        real_words.append(row[0])
real_dict = {w[0]:{'pron':w[1], 'pos':w[2]} for w in real_list}

real_words = set(real_words)

########################################################################################



DATA = 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(DATA)
model = BertForMaskedLM.from_pretrained(DATA)


"""
Do hotel managers get board with their jobs ?
Do hotel managers get [MASK] with their jobs ?
target: bored

I started to work at a seafood buffet , but then I pulled a mussel .
I started to work at a seafood buffet , but then I pulled a [MASK] .
target: muscle

I knew that the spirit couldn ' t float around very long . What ghost up must come down .
I knew that the spirit couldn ' t float around very long . What [MASK] up must come down .
target: goes
"""

masked_sent = [['Do hotel managers get [MASK] with their jobs ?', 'board', 'bored'],
                ["I started to work at a seafood buffet , but then I pulled a [MASK] .", 'mussel', 'muscle'],
                ["I knew that the spirit couldn ' t float around very long . What [MASK] up must come down .", 'ghost', 'goes']]

for sent, source, target in masked_sent:
    # Get the phonoemically similar words to the source word
    word1 = real_dict[source]['pron']
    options = []
    for entry in real_list:
        price = edit_distance(word1, entry[1])
        if price < 3 and source not in entry[0]:
            options.append(entry)

    print('Sentence: %s' %sent)
    print('source: %s' %source)
    print('target: %s\n' %target)
    loss_arr = []

    for option in options:
        # Replace [MASK] with option[0]
        sent_option = sent.replace('[MASK]', option[0])
        print(option[0])

        
        inputs = tokenizer(sent, return_tensors="pt")
        labels = tokenizer(sent_option, return_tensors="pt")["input_ids"][0][:inputs.input_ids.shape[1]]

        # mask labels of non-[MASK] tokens
        labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, torch.squeeze(labels, 0), -100)
        outputs = model(**inputs, labels=labels)
        #print(round(outputs.loss.item(), 2))
        loss_arr.append(outputs.loss.item())

    min_idx = torch.argmin(torch.tensor(loss_arr)).item()
    #print('\nPredicted:', options[min_idx][0])
    pred_sent = sent.replace('[MASK]', options[min_idx][0])
    print('Predicted sentence:', pred_sent)
    corrected_text = GingerIt().parse(pred_sent)
    print('Corrected sentence:', corrected_text['result'])
    print("-----------------------")