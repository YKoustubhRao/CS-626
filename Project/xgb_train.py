import sys
import cPickle
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from collections import Counter
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *

params_xgb = {
    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'eval_metric':'mlogloss',
    'num_class': 4
}
num_round = 1000


def cv():
    
    data_x, data_y = build_data()
    
    random_seed = 2017
    
    scores = []
    best_iters = [0]*5
    pscores = []
    with open('skf.pkl', 'rb') as infile:
        skf = cPickle.load(infile)

        for fold, (trainInd, validInd) in enumerate(skf.split(data_x, data_y)):
            print 'fold %s' % fold
            x_train = data_x[trainInd]
            y_train = data_y[trainInd]
            x_valid = data_x[validInd]
            y_valid = data_y[validInd]
            
            print 'perfect_score: ', perfect_score(y_valid)
            print Counter(y_valid)
            #break
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params_xgb, 
                            dtrain,
                            num_round,
                            watchlist,
                            verbose_eval=100)
            pred_y = bst.predict(dvalid)
            print pred_y
            print Counter(pred_y)
            print 'pred_y.shape'
            print pred_y.shape
            print 'y_valid.shape'
            print y_valid.shape
            s = fscore(pred_y, y_valid)
            s_perf = perfect_score(y_valid)
            print 'fold %s, score = %d, perfect_score %d' % (fold, s, s_perf)
            scores.append(s)
            pscores.append(s_perf)

    print 'scores:'
    print scores
    print 'mean score:'
    print np.mean(scores)

cv()