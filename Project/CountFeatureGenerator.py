import ngram
import cPickle
import pandas as pd
from nltk.tokenize import sent_tokenize
import hashlib


class CountFeatureGenerator(FeatureGenerator):


    def __init__(self, name='countFeatureGenerator'):
        super(CountFeatureGenerator, self).__init__(name)


    def process(self, df):

        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        print "generate counting features"
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        # overlapping n-grams count
        for gram in grams:
            df["count_of_Headline_%s_in_articleBody" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
            df["ratio_of_Headline_%s_in_articleBody" % gram] = \
                map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram])
        
        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]
        
        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]
        
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)
	    
        print 'BasicCountFeatures:'
        print df
        
        # split into train, test portion and save in separate files
        train = df[~df['target'].isnull()]
        print 'train:'
        print train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train = "train.basic.pkl"
        with open(outfilename_bcf_train, "wb") as outfile:
            cPickle.dump(feat_names, outfile, -1)
            cPickle.dump(xBasicCountsTrain, outfile, -1)
        print 'basic counting features for training saved in %s' % outfilename_bcf_train
        
        test = df[df['target'].isnull()]
        print 'test:'
        print test[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        #return 1
        if test.shape[0] > 0:
            # test set exists
            print 'saving test set'
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "test.basic.pkl"
            with open(outfilename_bcf_test, 'wb') as outfile:
                cPickle.dump(feat_names, outfile, -1)
                cPickle.dump(xBasicCountsTest, outfile, -1)
                print 'basic counting features for test saved in %s' % outfilename_bcf_test

        return 1


    def read(self, header='train'):

        filename_bcf = "%s.basic.pkl" % header
        with open(filename_bcf, "rb") as infile:
            feat_names = cPickle.load(infile)
            xBasicCounts = cPickle.load(infile)
            print 'feature names: '
            print feat_names
            print 'xBasicCounts.shape:'
            print xBasicCounts.shape
            #print type(xBasicCounts)

        return [xBasicCounts]

if __name__ == '__main__':

    cf = CountFeatureGenerator()
    cf.read()