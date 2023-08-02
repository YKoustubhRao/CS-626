# Run this only once
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("semcor") # downloads the .zip file, but doesn't unzip it
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download('omw-1.4')
nltk.download('brown')

# Better to download the tar.gz as the site keeps crashing "https://github.com/mmihaltz/word2vec-GoogleNews-vectors"
# import wget
# import gzip

# url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# filename = wget.download(url)

# f_in = gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb')
# f_out = open('GoogleNews-vectors-negative300.bin', 'wb')
# f_out.writelines(f_in)

# Import w2v here itself as it takes time to load
# from gensim.models import KeyedVectors
# W2V = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary = True)

from nltk.corpus import brown
import gensim
from gensim.models import Word2Vec
from helper import *

sentences = brown.sents()
v_size = 300
W2V  = gensim.models.Word2Vec(sentences, min_count = 1, vector_size = v_size, window = 5)
word_vectors = W2V.wv

from gensim.models import KeyedVectors

word_vectors.save('vectors.kv')
W2V = KeyedVectors.load('vectors.kv')


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

# Custom stopwords
EXTRA_SW = [
    "''",
    "'s",
    "``"
]

SW = stopwords.words("english")
SW += [p for p in punctuation]
SW += EXTRA_SW

lemmatizer = WordNetLemmatizer()

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

def parse(d):
    # d (nltk.corpus.reader.semcor.SemcorSentence) : can have lists as elements or nltk.tree.Tree

    tokens = []
    senses = []

    for e in d:

        if isinstance(e, nltk.tree.Tree):

            # e.label() returns a nltk.corpus.reader.wordnet.Lemma object or simply a string (of the form word.pos.num)
            lemma = e.label()
            
            if isinstance(lemma, nltk.corpus.reader.wordnet.Lemma):
                synset = lemma.synset() # nltk.corpus.reader.wordnet.Synset
                sense = syn2sense(synset)
            else:
                sense = None # ignore all tagged senses which aren't in WN (i.e. present as Lemma)
            
            le = len(e)
            if le == 1:
                w = e[0]
                if isinstance(w, nltk.tree.Tree) or isinstance(w, list):
                    # ignore w.label()
                    lw = len(w)
                    w = " ".join([w[i] for i in range(lw)])
            else:
                w = " ".join([e[i] for i in range(le)])

        elif isinstance(e, list):
            w = e[0]
            sense = None

        else:
            invtype = type(e)
            raise TypeError(f"Invalid type: {invtype}")

        if w:
            tokens.append(w)
            senses.append(sense)

    return tokens, senses

def getCandidate(w, tag):
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

    syns_b = wn.synsets(w, tag)
    syns = syns_b
    n_b = len(syns_b)

    for i in range(n_b):
        hyper = syns_b[i].hypernyms()
        syns.extend(hyper)
        n_h = len(hyper)
        for j in range(n_h):
            syns.extend(hyper[j].hyponyms())


    # For basic lesk
    # syns = wn.synsets(w, tag)



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

data = semcor.tagged_sents(tag = "sem") # 37176 sentences, 224716 tagged words, 34189 unique senses


n_total = 0
n_correct = 0
n_samples = 0

true = []
pred = []

#######################################################################################################################

# for d in data:

#     try:

#         tokens, senses = parse(d)
#         n_tokens = len(tokens)
#         print(n_samples)

#         # Tag and lemmatize tokens, don't remove stopwords here
#         tagged = nltk.pos_tag(tokens)
#         tags = [treebank2wn(p[1]) for p in tagged]
#         tokens = [lemmatize(w, tag) for w, tag in zip(tokens, tags)]

#         for i in range(n_tokens):

#             w = tokens[i]
#             tag = tags[i]
#             s_true = senses[i]

#             if not isinstance(w, str):
#                 raise TypeError(f"Invalid type: {type(w)} : {w} : {tokens}")

#             # Don't predict for words that aren't sense-tagged
#             if s_true is None:
#                 continue

#             # Get context for w (all words in the sentence except w)
#             context = tokens.copy()
#             del context[i] # more efficient than .pop(i)

#             # Remove stopwords and punctuation from context to reduce #elements in the context
#             # These don't contribute much to the semantic overlap anyways
#             cleaned = clean(context)
#             if len(cleaned) < 2:
#                 cleaned = context # if almost all words are stopwords, don't remove any

#             # Get context vector by average w2v vectors for each word
#             cv = sent2vec(cleaned)

#             if cv is None:
#                 # print(f"Empty context vector. Word: {w}, Cleaned: {cleaned}, Tokens: {tokens}. Using a random vector as context.")
#                 cv = np.random.rand(300,)

#             # Get WordNet candidate senses
#             sense_vectors, sense_labels = getCandidates(w, tag)
#             n_candidates = len(sense_labels)

#             s_pred = None
#             if n_candidates == 0:
#                 # Try without pos tag
#                 sense_vectors, sense_labels = getCandidates(w, None)
#                 n_candidates = len(sense_labels)
#                 if n_candidates == 0:
#                     # print(f"No synsets found. Word: {w}, Sense: {s_true}") # don't print, too many NE's in the data
#                     s_pred = random.choice(["group.n.01", "person.n.01", "location.n.01"]) # most likely an NE
            
#             # Use cosine similarity to get the best senses
#             best = -1 
#             for j in range(n_candidates):
#                 sv = sense_vectors[j]
#                 cs = cosineSimilarity(cv, sv)
#                 if cs > best:
#                     best = cs
#                     s_pred = sense_labels[j]

#             if s_true == s_pred:
#                 n_correct += 1
#             n_total += 1

#             true.append(s_true)
#             pred.append(s_pred)

#     except Exception as e:
#         print(f"Error at: {n_samples}")
#         print(str(e))
#         raise ValueError("Error")

#     n_samples += 1

#     if n_samples%200 == 0:
#         print(f"{n_samples} sentences processed")
#         acc = (n_correct/n_total)*100
#         print(f"Accuracy: {acc:.4f}")
#         print()


# pred_sense_set = set(pred)
# true_sense_set = set(true)
# all_senses = sorted(list(true_sense_set.union(pred_sense_set)))
# not_predicted = true_sense_set - pred_sense_set
# extra_predicted = pred_sense_set - true_sense_set

# acc = accuracy_score(true, pred)
# prec = precision_score(true, pred, average = "macro")
# rec = recall_score(true, pred, average = "macro")
# f1 = f1_score(true, pred, average = "macro")

# print(f"Accuracy: {acc:.4f}")
# print(f"Precision: {prec:.4f}")
# print(f"Recall: {rec:.4f}")
# print(f"F1-Score: {f1:.4f}")

######################################## PREDICT ##########################################

def predict(sent):

    senses = []
    tokens = word_tokenize(sent)
    # Tag and lemmatize tokens, don't remove stopwords here
    tagged = nltk.pos_tag(tokens)
    tags = [treebank2wn(p[1]) for p in tagged]
    tokens = [lemmatize(w, tag) for w, tag in zip(tokens, tags)]
    n_tokens = len(tokens)

    for i in range(n_tokens):

        w = tokens[i]
        tag = tags[i]

        # Get context for w (all words in the sentence except w)
        context = tokens.copy()
        del context[i] # more efficient than .pop(i)

        # Get context vector by average w2v vectors for each word
        cv = sent2vec(context)

        if cv is None:
            print(f"Empty context vector. Word: {w}, Tokens: {tokens}. Using a random vector as context.")
            cv = np.random.rand(300,)

        # Get WordNet candidate senses
        sense_vectors, sense_labels = getCandidates(w, tag)
        n_candidates = len(sense_labels)

        s_pred = None
        if n_candidates == 0:
            # Try without pos tag
            sense_vectors, sense_labels = getCandidates(w, None)
            n_candidates = len(sense_labels)
            if n_candidates == 0:
                print(f"No synsets found: {w}")
                s_pred = None

        # Use cosine similarity to get the best senses
        best = -1 
        for j in range(n_candidates):
            sv = sense_vectors[j]
            cs = cosineSimilarity(cv, sv)
            if cs > best:
                best = cs
                s_pred = sense_labels[j]

        senses.append(s_pred)

    return senses

# sents = [
#     "On combustion of coal we get ash"
#     # "The bank is located in the city near the river",
#     # "The stolen credit cards were found near the river bank",
#     # "The user had to kill the computer process",
#     # "The trees near Nuclear Power Plant were cut down"
# ]

print("You can input upto 5 sentences. If you want to input <5 sentences, please type \"Done\" when you are done!")
print("----------------------------")
i = 0
while i<5:
    sent = input('Enter a sentence: ')
    if sent == 'Done':
        break
    senses = predict(sent)
    for s in senses:
        if s is not None:
            print(s, ":", wn.synset(s).definition())
    print("----------------------------")
    i += 1