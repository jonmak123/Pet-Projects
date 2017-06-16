# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:55:38 2017

@author: Jonathan Mak
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
import time
import nltk
import random


##################################
###### Text processing part ######
def load_set(split):
    with open('pos.txt', 'r') as f:
        pos = f.readlines()
    
    with open('neg.txt', 'r') as f:
        neg = f.readlines()
    
    raw_set_x = pos + neg
    raw_y = [1] * len(pos) + [0] * len(neg)
    
    tokenizer = nltk.tokenize.RegexpTokenizer(r'(?!\d)+\w+')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    lines = [tokenizer.tokenize(x) for x in raw_set_x]    
    lines = [[lemmatizer.lemmatize(str(x)) for x in y] for y in lines]
    
    to_shuffle = list(zip(lines, raw_y))
    random.shuffle(to_shuffle)
    x_input,y_input = zip(*to_shuffle)
    
    y_input = [[1, 0] if k is 0 else [0, 1] for k in y_input]
    
    t_split = int(split * len(x_input))
    train_x = list(x_input[:t_split])
    train_y = np.array(list(y_input[:t_split]))
    test_x = list(x_input[t_split:])
    test_y = np.array(list(y_input[t_split:]))

    lexicon = list(set([w for l in lines for w in l]))
    max_length = max([len(x) for x in lines])
    
    return(lexicon, max_length, train_x, train_y, test_x, test_y)

def prepare_set(contents): 
    chunk = []
    for l in contents:
        line = []
        for word in l:
            vec = np.zeros(len(lexicon))
            vec[lexicon.index(word)] += 1
            line.append(vec)
        if len(line) < max_length:
            line += [[0] * len(lexicon)]*(max_length - len(line))
        chunk.append(line)
    return(np.array(chunk, 'float32'))
##################################


#############################
###### Tensorflow part ######

tf.reset_default_graph()

lexicon, max_length, train_x, train_y, test_x, test_y = load_set(0.9)

training_rate = 0.01

max_step = max_length
n_lex = len(lexicon)
n_hidden = 10
n_output = 2
batch_size = 20

x = tf.placeholder('float', [None, max_step, n_lex])
y = tf.placeholder('float', [None, n_output])

weight = tf.Variable(tf.truncated_normal([n_hidden, n_output]))
bias = tf.Variable(tf.truncated_normal([n_output]))

def RNN(x, weights = weight, biases = bias):
    x = tf.unstack(x, max_step, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 0.8)
    output, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
#    layer1 = output
#    layer1 = [tf.reshape(x, [-1]) for x in layer1]
#    layer1 = tf.stack(layer1)
#    layer1 = tf.transpose(layer1)
#    return(tf.matmul(layer1, weights) + biases)
    return(tf.matmul(output[-1], weights) + biases)

prediction = RNN(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = training_rate).minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def train_rnn():
    
    #start = time.time()
    init = tf.global_variables_initializer()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            sess.run(init)
    
            for epoch in range(5):
                train_index = 0
                test_index = 0
                train_acc = []
                test_acc = []
                
                while train_index < len(train_x):
                    print(train_index)
                    batch_x = prepare_set(train_x[train_index : train_index + batch_size])
                    batch_y = train_y[train_index : train_index + batch_size]
                    sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
                    acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
                    train_acc.append(acc)
                    train_index += batch_size
                print('epoch ', str(epoch+1) + ' training accuracy: ', sum(train_acc)/len(train_acc))
                
                while test_index < len(test_x):
                    batch_x = prepare_set(test_x[test_index : test_index + batch_size])
                    batch_y = test_y[test_index : test_index + batch_size]
                    acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
                    test_acc.append(acc)
                    test_index += batch_size
                print('epoch ', str(epoch+1) + ' testing accuracy: ', sum(test_acc)/len(test_acc))
    #end = time.time()
#    print('Total RNN runtime: ', int(end-start), 's')
#############################
    
    
    
    