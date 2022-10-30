import nltk
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from collections import Counter
from collections import defaultdict
from nltk.corpus import brown

# The following has to be performed only once
nltk.download('brown')
nltk.download('universal_tagset')

stime = time.time()

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']

words = list(brown.tagged_words(tagset='universal'))

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

def transition_probs(tag, words, tagcount):
    tokens, tag = zip(*words)
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
    transition_probability = transition_probs(tag, words, tag_count)

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

score_list = [0, 0, 0, 0, 0]
per_pos = [{}, {}, {}, {}, {}]
total_pred = np.empty([0,0])

tokens, tag = zip(*words)
total_length = len(words)

# Five fold cross validation
for i in range(5):
    x = int(i*total_length/5)
    y = int((i+1)*total_length/5)
    a = [(x.lower(), y) for x, y in words[:x]+words[y:]]
    b = [(key.lower(), val) for key, val in words[x:y]]
    score_list[i], per_pos[i], pred = Viterbi(a, b)
    total_pred = np.append(total_pred, np.array(pred))

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
cm_display.plot(xticks_rotation='vertical',ax=ax)
plt.savefig("img.png")

etime = time.time()

print(etime-stime)