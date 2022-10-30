import nltk
from nltk.corpus import brown
from collections import Counter
from collections import defaultdict
import gensim
from gensim.models import Word2Vec
import torch
from torch import nn
import numpy as np
import joblib

# The following has to be performed only once
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']
training_data = list(brown.tagged_words(tagset='universal'))
identity = torch.eye(12, dtype=torch.double)
tagset_dict = {'NOUN':identity[0,:], 'VERB':identity[1,:], '.':identity[2,:], 'ADP':identity[3,:], 'DET':identity[4,:], 'ADJ':identity[5,:], 'ADV':identity[6,:], 'PRON':identity[7,:], 'CONJ':identity[8,:], 'PRT':identity[9,:], 'NUM':identity[10,:], 'X':identity[11,:]}

# Word vector size for HMM-Viterbi-Vector and FFNN-BP
v_size_v = 16
v_size_nn = 128

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

def Viterbi(test, tag_freq, emission_probability, transition_probability):

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
            em = emission_probability[test[i]][tag_i]
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

        print(test[i], ":", max_tag, sep="", end=" ")

        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%8 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    return

def Viterbi_vect(test, data_w, data_t, tag_freq, emission_probability, transition_probability):

    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size_v, window = 5, workers = 4)
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size_v, window = 5, workers = 4)

    viterbi = {}
    for tag_i in tagset:
      viterbi[tag_i] = transition_probability["."][tag_i]

    k = 1

    n = len(test)
    for i in range(n):

        maxi = 0.0
        max_tag = "tag"
        c = False
        copy = test[i]

        for tag_i in tagset:
            em = emission_probability[test[i][0]][tag_i]

            if em != 0:
                c = True

        if not c:
            test[i] = model_train.wv.similar_by_vector(model_test.wv[test[i]], topn=1)[0][0]

        viterbi_h = {}
        for tag_i in tagset:
            em = emission_probability[test[i]][tag_i]
            val = viterbi[tag_i]*em*100
            viterbi_h[tag_i] = val
            
            if val >= maxi:
                maxi = val
                max_tag = tag_i

        print(copy, ":", max_tag, sep="", end=" ")

        viterbi_next = dict.fromkeys(tagset, 0)
        for tag_i in tagset:
            for tag_j in tagset:
                viterbi_next[tag_j] = max(viterbi_next[tag_j], viterbi_h[tag_i]*transition_probability[tag_i][tag_j])

        if k%8 == 0:
            k = 0
            viterbi = tag_freq.copy()
        
        else:
            viterbi = viterbi_next

        k = k+1

    return

class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
            self.l1 = nn.Linear(v_size_nn, 256)
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

def FFNN(data_t):
    
    model_train = Word2Vec.load('models/word2vec.model')
    model_test = gensim.models.Word2Vec(data_t, min_count = 1, vector_size = v_size_nn, window = 5, workers = 4)

    saved_model = joblib.load('models/test_model.pkl')

    pred = []
    im_tired = []

    for tok in data_t[0]:
      try:
        model_train.wv[tok]
      except:
        im_tired.append(model_test.wv[tok])
      else:
        im_tired.append(model_train.wv[tok])

    i_am_tired = np.array(im_tired)
    predict = saved_model(torch.from_numpy(i_am_tired))
    index = torch.argmax(predict, 1)
    for ind in index:
        pred.append(tagset[ind])
      
    for i in range(len(data_t[0])):
        print(data_t[0][i], ":", pred[i], sep="", end=" ")
        
    return

a = [(key.lower(), val) for key, val in training_data]
tokens, tag = zip(*a)

taglist = torch.empty((len(a),12),dtype=torch.float64)
for i in range(len(a)):
  taglist[i] = tagset_dict[a[i][1]].unsqueeze(0)

tag_count, tag_freq = tag_frequency(tag, a)
emission_probability = emission_probs(tokens, a, tag_count)
transition_probability = transition_probs(tag, tag_count)

data_w = []
temp_w = []
for j in range(len(a)):
    temp_w.append(a[j][0])
    if a[j][0] == ".":
        data_w.append(temp_w)
        temp_w = []
data_w.append(temp_w)

i = 0
print("\nNote: In one run of the code, you can input a maximum of 5 sentences. If you want to input <5 sentences, please type \"Done\" when you are done!", end="")
while i<5:
    print("\n")
    user_input = input('Enter a sentence: ')
    if user_input == 'Done':
        break
    words = nltk.word_tokenize(user_input)
    b = [key.lower() for key in words]

    print("\n########### HMM VITERBI SYMBOLIC ###########")
    Viterbi(b, tag_freq, emission_probability, transition_probability)

    data_t = []
    temp_t=[]
    for j in range(len(b)):
        temp_t.append(b[j])
        if b[j] == ".":
            data_t.append(temp_t)
            temp_t = []
    data_t.append(temp_t)

    print("\n\n############ HMM VITERBI VECTOR ############")
    Viterbi_vect(b, data_w, data_t, tag_freq, emission_probability, transition_probability)

    print("\n\n################ FFNN AND BP ################")
    FFNN(data_t)

    i = i + 1