import nltk
import time
from collections import Counter
from collections import defaultdict
from nltk.corpus import brown

# The following has to be performed only once
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')

stime = time.time()

tagset = ['NOUN', 'VERB', '.', 'ADP', 'DET', 'ADJ', 'ADV', 'PRON', 'CONJ', 'PRT', 'NUM', 'X']

training_data = list(brown.tagged_words(tagset='universal'))

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

        print(test[i] + ": " + max_tag)


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

a = [(key.lower(), val) for key, val in training_data]
tokens, tag = zip(*a)
tag_count, tag_freq = tag_frequency(tag, a)
emission_probability = emission_probs(tokens, a, tag_count)
transition_probability = transition_probs(tag, a, tag_count)

i = 0
print("In one run of the code, you can input a maximum of 5 sentences. If you want to input <5 sentences, please type \"Done\" when you are done!")
while i<5:
    user_input = input('Enter a sentence: ')
    if user_input == 'Done':
        break
    words = nltk.word_tokenize(user_input)
    b = [key.lower() for key in words]
    Viterbi(b, tag_freq, emission_probability, transition_probability)
    i = i + 1

etime = time.time()

print(etime-stime)