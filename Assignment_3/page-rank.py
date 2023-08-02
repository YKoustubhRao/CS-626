from nltk.corpus import wordnet as wn 
import networkx as nx
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from helperFUn import leskSim, edge_weight, graph, senseAssignment
from sys import argv
import matplotlib.pyplot as plt

sent = input('Enter a sentence: ')

print("----------------------------")

G = graph(sent)  # "my country india"

ranks = nx.pagerank(G,alpha=0.4)
tokenizer = RegexpTokenizer(r'\w+')
s1 = tokenizer.tokenize("my country india")
dict1 = {}
for i in range(len(s1)):
    dict1[i] = [str(k) for k in wn.synsets(s1[i])]  #all the word senses of i th word here
    dict1[i] = [re.findall(r"'(.*?)'",o)[0] for o in dict1[i]]

senseLst = senseAssignment(dict1,ranks)
for i in range(len(dict1)):
    print (s1[i],":",senseLst[i])

print("----------------------------")

#senses corresponding to each of the word
for i in range(len(dict1)):
    print (s1[i],":",dict1[i])

print("----------------------------")

print(ranks)

edges = [(u, v) for (u, v, d) in G.edges(data=True)]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=60)

# edges
nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)

nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')

plt.axis('off')
plt.savefig('Network.png')
plt.show()
