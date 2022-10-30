import nltk
from nltk.corpus import brown
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from collections import Counter
from collections import defaultdict
import gensim
import torch
from torch import nn

# The following has to be performed only once
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

stime = time.time()

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']
identity = torch.eye(12, dtype=torch.double)
tagset_dict = {'NOUN':identity[0,:], 'VERB':identity[1,:], '.':identity[2,:], 'ADP':identity[3,:], 'DET':identity[4,:], 'ADJ':identity[5,:], 'ADV':identity[6,:], 'PRON':identity[7,:], 'CONJ':identity[8,:], 'PRT':identity[9,:], 'NUM':identity[10,:], 'X':identity[11,:]}
words = list(brown.tagged_words(tagset='universal'))
v_size = 128 # Word Vector Size

def tag_frequency(tag, words):
    total = len(words)
    tagcount = Counter(tag) 
    tagfreq = {}

    for i in tagcount.keys():
        tagfreq[i] = tagcount[i]/total
        
    return tagcount, tagfreq

def emission_probs(tokens, words, tagcount):

    tokenTags = defaultdict(Counter)
    
    for token_i, tag_i in words:
        tokenTags[token_i][tag_i] += 1
    
    for tag_i in tagset:
        for token_i in tokens:
            if tokenTags[token_i][tag_i] >= 1:
                tokenTags[token_i][tag_i] = tokenTags[token_i][tag_i]/(tagcount[tag_i])

    return tokenTags

def transition_probs(tag, tagcount):
    pl = len(tag) - 1
    tagtags = defaultdict(Counter)

    for i in range(pl-1):
        tagtags[tag[i]][tag[i+1]] += 1

    for tag_i in tagset:
        for tag_j in tagset:
            tagtags[tag_i][tag_j] = tagtags[tag_i][tag_j]/(tagcount[tag_i])

    return tagtags

def Viterbi(words, test): 

    tokens, tag = zip(*words)
    tag_count, tag_freq = tag_frequency(tag, words)
    emission_probability = emission_probs(tokens, words, tag_count)
    transition_probability = transition_probs(tag, tag_count)

    per_pos = dict.fromkeys(tag_freq.keys(), 0.0)
    per_pos_count = dict.fromkeys(tag_freq.keys(), 0)
    pred = []

    score = 0.0
    viterbi = {}
    for tag_i in tagset:
      viterbi[tag_i] = transition_probability["."][tag_i]

    k = 1
    n = len(test)

    for i in range(n):

        maxi = 0.0
        max_tag = "tag"
        c = False

        viterbi_h = {}
        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]
            val = viterbi[tag_i]*em*100
            viterbi_h[tag_i] = val
            
            if val >= maxi:
                maxi = val
                max_tag = tag_i

            if em != 0:
                c = True

        if not c:
            maxi = 0.0
            max_tag = "tag"
            for tag_i in tagset:
                val = viterbi[tag_i]*100*(1/(1+tag_count[tag_i]))
                viterbi_h[tag_i] = val
                if val >= maxi:
                    maxi = val
                    max_tag = tag_i

        per_pos_count[test[i][1]] = per_pos_count[test[i][1]] + 1
        pred.append(max_tag)

        if max_tag == test[i][1]:
            per_pos[test[i][1]] = per_pos[test[i][1]] + 1.0
            score = score + 1.0

        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%7 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    for tag_i in tagset:
        per_pos[tag_i] = per_pos[tag_i]/per_pos_count[tag_i]

    return score*100/n, per_pos, pred

def Viterbi_vec(words, test,data_w,data_t): 

    tokens, tag = zip(*words)

    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5, workers = 4)
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size, window = 5, workers = 4)
    
    tag_count, tag_freq = tag_frequency(tag, words)
    emission_probability = emission_probs(tokens, words, tag_count)
    transition_probability = transition_probs(tag, tag_count)

    per_pos = dict.fromkeys(tag_freq.keys(), 0.0)
    per_pos_count = dict.fromkeys(tag_freq.keys(), 0)
    pred = []

    score = 0.0
    viterbi = {}
    for tag_i in tagset:
      viterbi[tag_i] = transition_probability["."][tag_i]

    k = 1
    n = len(test)

    for i in range(n):

        maxi = 0.0
        max_tag = "tag"
        c = False

        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]

            if em != 0:
                c = True

        if not c:
            lst = list(test[i])
            lst[0] = model_train.wv.similar_by_vector(model_test.wv[test[i][0]], topn=1)[0][0]
            test[i] = tuple(lst)

        viterbi_h = {}
        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]
            val = viterbi[tag_i]*em*100
            viterbi_h[tag_i] = val
            
            if val >= maxi:
                maxi = val
                max_tag = tag_i

        per_pos_count[test[i][1]] = per_pos_count[test[i][1]] + 1
        pred.append(max_tag)

        if max_tag == test[i][1]:
            per_pos[test[i][1]] = per_pos[test[i][1]] + 1.0
            score = score + 1.0

        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%7 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    for tag_i in tagset:
        per_pos[tag_i] = per_pos[tag_i]/per_pos_count[tag_i]

    return score*100/n, per_pos, pred

def FFNN(words, test,data_w, data_t, taglist, taglist_t):
    tokens, tag = zip(*words)
    t_tokens, t_tag = zip(*test)
    
    n_t = len(test)

    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5, workers = 4)
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size, window = 5, workers = 4)

    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.l1 = nn.Linear(v_size,256)
            self.l2 = nn.Linear(256,512)
            self.l3 = nn.Linear(512,512)
            self.l4 = nn.Linear(512,12)
            self.relu = nn.ReLU()
        
        def forward(self,x):
            x = self.l1(x) 
            x = self.relu(x)
            x = self.l2(x) 
            x = self.relu(x)
            x = self.l3(x) 
            x = self.relu(x)
            x = self.l4(x) 
            output = x
            return output.double()

    model = net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 200
    model.zero_grad()
    n_train = len(tokens)//1000
    bad_ep = 0
    prev_cost = 10.0

    for _ in range(epochs):
        model.train()
        for i in range(1000):
          y_pred = model(torch.from_numpy(model_train.wv[tokens[i*n_train:(i+1)*n_train]]))
          cost = criterion(y_pred, taglist[i*n_train:(i+1)*n_train])
          optimizer.zero_grad()
          cost.backward()
          optimizer.step()
        y_pred = model(torch.from_numpy(model_train.wv[tokens]))
        cost = criterion(y_pred, taglist)
        if cost >= prev_cost:
            bad_ep += 1
        else:
            bad_ep = 0
        if bad_ep >= 5:
            break
        prev_cost = cost
        print("cost:", cost)


    y_pred = model(torch.from_numpy(model_train.wv[tokens]))
    cost = criterion(y_pred, taglist)
    rando, y_pred_tags = torch.max(y_pred, dim = 1) 
    y_pred_tags = torch.eye(12)[y_pred_tags,:]
    print(y_pred_tags.shape)
    print(taglist.shape)
    correct_pred = ((y_pred_tags == taglist)*(y_pred_tags == 1)).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    print("accuracy:", acc)
    print("cost:", cost)


    per_pos = torch.zeros((12,1),dtype=torch.float64)
    per_pos_count = torch.zeros((12,1),dtype=torch.int64)

    pred = []
    score = 0.0
    im_tired = []

    for tok in t_tokens:
      try:
        model_train.wv[tok]
      except:
        im_tired.append(model_test.wv[tok])
      else:
        im_tired.append(model_train.wv[tok])

    i_am_tired = np.array(im_tired)
    predict = model(torch.from_numpy(i_am_tired))

    index = torch.argmax(predict,1)
    for ind in index:
      pred.append(tagset[ind])
    per_pos_count = torch.count_nonzero(taglist_t,dim=0)
    for i in range(n_t):
      per_pos[index[i]] += taglist_t[i,index[i]]
      score = score + taglist_t[i,index[i]]
    per_pos = per_pos/per_pos_count

    return score*100/n_t, per_pos, pred

score_list_v = [0, 0, 0, 0, 0]
per_pos_v = [{}, {}, {}, {}, {}]
total_pred_v = np.empty([0,0])
score_list_nn = [0, 0, 0, 0, 0]
per_pos_nn = [{}, {}, {}, {}, {}]
total_pred_nn = np.empty([0,0])
score_list = [0, 0, 0, 0, 0]
per_pos = [{}, {}, {}, {}, {}]
total_pred = np.empty([0,0])

tokens, tag = zip(*words)
total_length = len(words)
taglist = torch.empty((total_length,12), dtype=torch.float64)
for i in range(total_length):
  taglist[i] = tagset_dict[words[i][1]].unsqueeze(0)

data_t = [[],[],[],[],[]]
data_w = [[],[],[],[],[]]

for i in range(5):
    temp_w = []
    temp_t=[]
    x = int(i*total_length/5)
    y = int((i+1)*total_length/5)
    tokens_w = [l.lower() for l in tokens[:x]+tokens[y:]]
    tokens_t = [l.lower() for l in tokens[x:y]]
    for j in range(len(tokens_w)):
        temp_w.append(tokens_w[j])
        if tokens_w[j] == ".":
            data_w[i].append(temp_w)
            temp_w = []
    data_w[i].append(temp_w)
    for j in range(len(tokens_t)):
        temp_t.append(tokens_t[j])
        if tokens_t[j] == ".":
            data_t[i].append(temp_t)
            temp_t = []
    data_t[i].append(temp_t)

# Five fold cross validation
for i in range(5):
    x = int(i*total_length/5)
    y = int((i+1)*total_length/5)
    a = [(x.lower(), y) for x, y in words[:x]+words[y:]]
    b = [(key.lower(), val) for key, val in words[x:y]]

    score_list[i], per_pos[i], pred = Viterbi(a, b)
    total_pred = np.append(total_pred, np.array(pred))

    score_list_v[i], per_pos_v[i], pred_v = Viterbi_vec(a, b,data_w[i],data_t[i])
    total_pred_v = np.append(total_pred_v, np.array(pred_v))

    score_list_nn[i], per_pos_nn[i], pred_nn = FFNN(a, b,data_w[i],data_t[i],torch.cat((taglist[:x,:],taglist[y:,:])),taglist[x:y])
    total_pred_nn = np.append(total_pred_nn, np.array(pred_nn))


print("---------------------------------------- Viterbi Symbolic Evaluation ----------------------------------------------------")

print("Scores list: ", score_list, '\t', 'Avg: ', np.average(score_list), '\n')
print("Per POS Accuracy: ")
print(per_pos[0])
print(per_pos[1])
print(per_pos[2])
print(per_pos[3])
print(per_pos[4], '\n')

var = metrics.precision_recall_fscore_support(tag, total_pred, average=None, labels=tagset, zero_division=0)
print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred, average='weighted', labels=tagset, zero_division=0)[0], '\n')
print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred, average='weighted', labels=tagset, zero_division=0)[1], '\n')

f100 = metrics.fbeta_score(tag, total_pred, average=None, beta=1, labels=tagset, zero_division=0)
print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
f50 = metrics.fbeta_score(tag, total_pred, average=None, beta=0.5, labels=tagset, zero_division=0)
print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
f200 = metrics.fbeta_score(tag, total_pred, average=None, beta=2, labels=tagset, zero_division=0)
print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')

confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred))
np.set_printoptions(precision=2)
confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(xticks_rotation='vertical', ax=ax)
plt.savefig('figures/img_Viterbi.png')


print("---------------------------------------- Viterbi Vector Evaluation ----------------------------------------------------")

print("Scores list: ", score_list_v, '\t', 'Avg: ', np.average(score_list_v), '\n')
print("Per POS Accuracy: ")
print(per_pos_v[0])
print(per_pos_v[1])
print(per_pos_v[2])
print(per_pos_v[3])
print(per_pos_v[4], '\n')

var = metrics.precision_recall_fscore_support(tag, total_pred_v, average=None, labels=tagset, zero_division=0)
print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_v, average='weighted', labels=tagset, zero_division=0)[0], '\n')
print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_v, average='weighted', labels=tagset, zero_division=0)[1], '\n')

f100 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=1, labels=tagset, zero_division=0)
print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
f50 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=0.5, labels=tagset, zero_division=0)
print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
f200 = metrics.fbeta_score(tag, total_pred_v, average=None, beta=2, labels=tagset, zero_division=0)
print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_v, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')

confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred_v))
np.set_printoptions(precision=2)
confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(xticks_rotation='vertical', ax=ax)
plt.savefig('figures/img_Viterbi_vec.png')


print("------------------------------------------ FFNN-BP Evaluation ----------------------------------------------------")

print("Scores list: ", score_list_nn, '\t', 'Avg: ', np.average(score_list_nn), '\n')
print("Per POS Accuracy: ")
print(per_pos_nn[0])
print(per_pos_nn[1])
print(per_pos_nn[2])
print(per_pos_nn[3])
print(per_pos_nn[4], '\n')

var = metrics.precision_recall_fscore_support(tag, total_pred_nn, average=None, labels=tagset, zero_division=0)
print("Precision: ", var[0], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_nn, average='weighted', labels=tagset, zero_division=0)[0], '\n')
print("Recall: ", var[1], '\t', 'Weighted Avg: ', metrics.precision_recall_fscore_support(tag, total_pred_nn, average='weighted', labels=tagset, zero_division=0)[1], '\n')

f100 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=1, labels=tagset, zero_division=0)
print("F1-score: ", f100, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=1, labels=tagset, zero_division=0), '\n')
f50 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=0.5, labels=tagset, zero_division=0)
print("F0.5-score: ", f50, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=0.5, labels=tagset, zero_division=0), '\n')
f200 = metrics.fbeta_score(tag, total_pred_nn, average=None, beta=2, labels=tagset, zero_division=0)
print("F2-score: ", f200, '\t', 'Weighted Avg: ', metrics.fbeta_score(tag, total_pred_nn, average='weighted', beta=2, labels=tagset, zero_division=0), '\n')

confusion_matrix = np.transpose(metrics.confusion_matrix(tag, total_pred_nn))
np.set_printoptions(precision=2)
confusion_matrix = np.round_(np.transpose(confusion_matrix/np.sum(confusion_matrix,0)),decimals=2)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = tagset)
fig, ax = plt.subplots(figsize=(10,10))
cm_display.plot(xticks_rotation='vertical', ax=ax)
plt.savefig('figures/img_FFNN.png')

etime = time.time()

print(etime-stime)