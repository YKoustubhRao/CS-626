from util import *
import random
import tensorflow as tf
import pandas as pd
import csv

print("--------- HEADLINE ---------")
headline_input = input('Enter Headline: ')

i = 0
print("--------- BODY ---------")
print("You can input upto 10 paras. If you want to input <10 paras, please type \"Done\" when you are done!")
body_input = ""
while i<10:
    x = input('Enter a para in Body: ')
    if x == 'Done':
        break
    body_input = body_input + " " + x

header1 = ['Body ID', 'articleBody']
data1 = [11032003, body_input]
with open('data/demo_bodies.csv', 'w', encoding='UTF8', newline='') as f1:
    writer = csv.writer(f1)
    writer.writerow(header1)
    writer.writerow(data1)
f1.close()

header2 = ['Headline', 'Body ID']
data2 = [headline_input, 11032003]
with open('data/demo_stances_unlabeled.csv', 'w', encoding='UTF8', newline='') as f2:
    writer = csv.writer(f2)
    writer.writerow(header2)
    writer.writerow(data2)
f2.close()

# Set file names
file_train_instances = "data/train_stances.csv"
file_train_bodies = "data/train_bodies.csv"
file_test_instances = "data/demo_stances_unlabeled.csv"
file_test_bodies = "data/demo_bodies.csv"

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90

# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

# Create placeholders
features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.arg_max(softmaxed_logits, 1)

with tf.Session() as sess:
    load_model(sess)

    # Predict
    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    test_pred = sess.run(predict, feed_dict=test_feed_dict)

pred = None
for i in test_pred:
    pred = label_ref_rev[i]

print("--------- PREDICTION ---------")
print("Headline: ", headline_input)
print("Body: ", body_input)
print("Stance: ", pred)