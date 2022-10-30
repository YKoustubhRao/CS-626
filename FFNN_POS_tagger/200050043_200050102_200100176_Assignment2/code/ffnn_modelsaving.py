import numpy as np
import joblib
import nltk
from nltk.corpus import brown
import gensim
import torch
from torch import nn

# The following has to be performed only once
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']
training_data = list(brown.tagged_words(tagset='universal'))
v_size = 128
identity = torch.eye(12, dtype=torch.double)
tagset_dict = {'NOUN':identity[0,:], 'VERB':identity[1,:], '.':identity[2,:], 'ADP':identity[3,:], 'DET':identity[4,:], 'ADJ':identity[5,:], 'ADV':identity[6,:], 'PRON':identity[7,:], 'CONJ':identity[8,:], 'PRT':identity[9,:], 'NUM':identity[10,:], 'X':identity[11,:]}

words = list(brown.tagged_words(tagset='universal'))

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

def FFNN(words, data_w,  taglist):

    tokens, tag = zip(*words)
    
    model_train = gensim.models.Word2Vec(data_w, min_count = 1, vector_size = v_size, window = 5, workers = 4)
    model_train.save("word2vec.model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    model.zero_grad()
    n_train = len(tokens)//1000
    for epo in range(epochs):
        model.train()
        for i in range(1000):
            y_pred = model(torch.from_numpy(model_train.wv[tokens[i*n_train:(i+1)*n_train]]))
            cost = criterion(y_pred, taglist[i*n_train:(i+1)*n_train])
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        y_pred = model(torch.from_numpy(model_train.wv[tokens]))
        cost = criterion(y_pred, taglist)
        print("cost:", cost, "; epoch:", epo)


    y_pred = model(torch.from_numpy(model_train.wv[tokens]))
    cost = criterion(y_pred, taglist)
    rando, y_pred_tags = torch.max(y_pred, dim = 1) 
    y_pred_tags = torch.eye(12)[y_pred_tags,:]
    # print(y_pred_tags.shape)
    # print(taglist.shape)
    correct_pred = ((y_pred_tags == taglist)*(y_pred_tags == 1)).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    print("accuracy:", acc)
    print("cost:", cost)

    pred = []
    im_tired = []

    for tok in tokens:
        im_tired.append(model_train.wv[tok])  

    i_am_tired = np.array(im_tired)
    predict = model(torch.from_numpy(i_am_tired))
    joblib.dump(model, 'test_model.pkl')
    index = torch.argmax(predict,1)
    for ind in index:
      pred.append(tagset[ind])

    acc = 0.0
    print(predict[0])

    for i in range(len(tokens)):
      if pred[i]==tokens[i]:
        acc=acc+1
    acc = acc/len(words)
    return acc

total_length = len(words)
taglist = torch.empty((total_length,12), dtype=torch.float64)
for i in range(total_length):
    taglist[i] = tagset_dict[words[i][1]].unsqueeze(0)

a = [(x.lower(), y) for x, y in words]
data_w = []
temp_w = []
for j in range(total_length):
    temp_w.append(a[j][0])
    if a[j][0] == ".":
        data_w.append(temp_w)
        temp_w = []
data_w.append(temp_w)

FFNN(a, data_w, taglist)