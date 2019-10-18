#!/usr/bin/env python
# encoding: utf-8
'''
source: https://github.com/aymericdamien/TensorFlow-Examples
或者见tensorflow github tutorials examples: https://github.com/tensorflow/tensorflow
'''


from __future__ import division, print_function, absolute_import

import collections
import os
import random
# import urllib
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf


# Training Parameters
learning_rate = 0.1
batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# Evaluation Parameters
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec Parameters
embedding_size = 200 # Dimension of the embedding vector
max_vocabulary_size = 50000 # Total number of different words in the vocabulary
min_occurrence = 10 # Remove all words that does not appears at least n times
skip_window = 3 # How many words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label
num_sampled = 64 # Number of negative examples to sample



def download_data(data_path):
    # Download a small chunk of Wikipedia articles collection
    url = 'http://mattmahoney.net/dc/text8.zip'
    # if not os.path.exists(data_path):
    #     print("Downloading the dataset... (It may take some time)")
    #     filename, _ = urllib.request.urlretrieve(url, data_path)
    #     print("Done!")

def read_data(data_path):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(data_path) as f:
        text_words = f.read(f.namelist()[0]).lower().split()
    return text_words

def build_dataset(words):
    '''
    下面是TF的word2vec要求的做法，目的是实现词频越大，词的类别编号也就越大。
    也间接说明在TF的word2vec里，负采样的过程其实就是优先采词频高的词作为负样本。

    @:return
    text_words 是corpus中的所有词
    count是top词频max_vocabulary_size = 50000的词，并且去除了min_occurrence = 10的词
    word2id给count中的词加个顺序编号
    data等于是把text_words中的词全部转成Index，如果不在，Index=0
    id2word是把word2id反过来
    '''
    # Build the dictionary and replace rare words with UNK token
    count = [('UNK', -1)]
    # Retrieve the most common words
    count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))  # 计数然后取前*个
    # Remove samples with less than 'min_occurrence' occurrences
    for i in range(len(count) - 1, -1, -1):
        if count[i][1] < min_occurrence:
            count.pop(i)
        else:
            # The collection is ordered, so stop when 'min_occurrence' is reached
            break
    # Compute the vocabulary size
    vocabulary_size = len(count)
    # Assign an id to each word
    word2id = dict()
    for i, (word, _) in enumerate(count):
        word2id[word] = i

    data = list()
    unk_count = 0
    for word in text_words:
        # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary
        index = word2id.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return  data, count, vocabulary_size, word2id, id2word


# -----------    step 1 prepare input data and build vocabulary
data_path = 'wikipedia/text8.zip'
# download_data(data_path)
text_words = read_data(data_path)
data, count, vocabulary_size, word2id, id2word = build_dataset(text_words)
print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])



# ------------   Step 2: Function to generate a training batch for the skip-gram model.
data_index = 0
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

 # -------------   Step 3: Build and train a skip-gram model.
# Input data
X = tf.placeholder(tf.int32, shape=[None])
# Input label
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Ensure the following ops & var are assigned on CPU
# (some ops are not compatible on GPU)
with tf.device('/cpu:0'):
    # Create the embedding variable (each row represent a word embedding vector)
    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    # Lookup the corresponding embedding vectors for each sample in X
    X_embed = tf.nn.embedding_lookup(embedding, X)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch, 下面是tf word2vec的固定写法
# tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
loss_op = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=Y,
                   inputs=X_embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

# Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation
# Compute the cosine similarity between the minibatch samples embedding and all embedding
X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))
embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))
cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Testing data
    x_test = np.array([word2id[bytes(w, encoding='utf8')] for w in eval_words]) # 因为word2id中是从wiki数据读进来的，是bytes格式

    average_loss = 0
    for step in range(1, num_steps + 1):
        # Get a new batch of data
        batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
        # Run training op
        _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        average_loss += loss

        if step % display_step == 0 or step == 1:
            if step > 1:
                average_loss /= display_step
            print("Step " + str(step) + ", Average Loss= " + \
                  "{:.4f}".format(average_loss))
            average_loss = 0

        # Evaluation
        if step % eval_step == 0 or step == 1:
            print("Evaluation...")
            sim = sess.run(cosine_sim_op, feed_dict={X: x_test})
            for i in range(len(eval_words)):
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = '"%s" nearest neighbors:' % eval_words[i]
                for k in range(top_k):
                    log_str = '%s %s,' % (log_str, id2word[nearest[k]])
                print(log_str)