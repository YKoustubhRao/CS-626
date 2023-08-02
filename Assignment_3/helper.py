import nltk
from nltk import word_tokenize
from nltk.corpus import semcor # corpus reader: https://github.com/nltk/nltk/blob/develop/nltk/corpus/reader/semcor.py
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import random
import numpy as np
from tqdm.notebook import tqdm
from string import punctuation
from num2words import num2words
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors


lemmatizer = WordNetLemmatizer()
W2V = KeyedVectors.load('vectors.kv')

EXTRA_SW = [
    "''",
    "'s",
    "``"
]

SW = stopwords.words("english")
SW += [p for p in punctuation]
SW += EXTRA_SW

def cosineSimilarity(a, b):
    cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cs


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def n2w(w):
    # converts given n to word form if n is numeric
    if isNumber(w) and w.lower() != "infinity" and w.lower() != "nan":
        w = num2words(w)
    return w


def lemmatize(w, tag):
    if tag is None:
        return lemmatizer.lemmatize(w)
    else:
        return lemmatizer.lemmatize(w, tag)


def clean(tokens):
    tagged = nltk.pos_tag(tokens)
    lemmatized = [lemmatize(w, treebank2wn(tag)) for w, tag in tagged]
    cleaned = [n2w(w) for w in lemmatized if w.lower() not in SW]
    return cleaned

def getVec(w):
    # Returns (300,) shaped numpy array
    try:
        v = W2V[w]
        return v
    except KeyError:
        return None # ignore words not in vocab

def syn2sense(syn):
    # get the sense (= lemma.postag.num) for a given synset
    s = syn.name()
    # n = ".".join(s.split(".")[-2:]) # n.01 and v.01 are different senses (eg: ash.n.01, ash.v.01)
    return s

def treebank2wn(ttag):
    if ttag.startswith("J"):
        return wn.ADJ
    elif ttag.startswith("V"):
        return wn.VERB
    elif ttag.startswith("N"):
        return wn.NOUN
    elif ttag.startswith("R"):
        return wn.ADV
    else:
        return None

def sent2vec(tokens):

    v = 0
    n = 0

    for w in tokens:

        # Check if w is a named entity (TODO: use wordnet NE tag directly instead of below approach)
        tkns = word_tokenize(w)

        if len(tkns) > 1:
            for t in tkns:
                vt = getVec(t)
                if vt is not None:
                    n += 1
                    v += vt
        else:
            vw = getVec(w)
            if vw is not None:
                n += 1
                v += vw

    if n == 0: # when tokens is empty or no token in word2vec
        v = None
    else:
        v /= n

    return v


def getCandidates(w, tag):
    # Get candidate sense vectors and labels of a word w

    w = w.replace(".", "") # "Sept." becomes "Sept"
    w = w.replace("-", "") # re-elected becomes "reelected"

    # Handle ngrams (like "united states")
    tkns = word_tokenize(w)
    if len(tkns) > 1:
        tagged = nltk.pos_tag(tkns)
        tags = [treebank2wn(p[1]) for p in tagged]
        ltkns = [lemmatize(w, t) for w, t in zip(tkns, tags)]
        w = "_".join(ltkns)


    # Only Hypernyms and Hyponyms are taken, if you want add more. Extended Lesk

    # syns_b = wn.synsets(w, tag)
    # syns = syns_b
    # n_b = len(syns_b)

    # for i in range(n_b):
    #     hyper = syns_b[i].hypernyms()
    #     syns.extend(hyper)
    #     n_h = len(hyper)
    #     for j in range(n_h):
    #         syns.extend(hyper[j].hyponyms())


    # For basic lesk
    syns = wn.synsets(w, tag)



    if len(syns) == 0:
        w = "_".join(tkns) # cases where lemmatization doesn't help ("agreed upon")
        syns = wn.synsets(w, tag)

    sense_vectors = []
    sense_labels = []

    for syn in syns:

        label = syn2sense(syn)

        defn = syn.definition()

        defn = defn.replace("_", " ")
        defn = defn.replace("-", " ")

        tkns = word_tokenize(defn)
        if len(tkns) == 0:
            raise ValueError(f"0 tokens found: {defn}")

        clnd = clean(tkns)
        if len(clnd) < 2:
            clnd = tkns # don't remove stopwords if the sentence is almost entirely made up of them

        sv = sent2vec(clnd)

        if sv is None:
            # print(f"Empty sense vector. Word: {w}, Definition: {defn}, Cleaned: {clnd}. Using a random vector as sense.")
            sv = np.random.rand(300,)
        
        sense_vectors.append(sv)
        sense_labels.append(label)

    return sense_vectors, sense_labels # returns empty lists if no synsets found