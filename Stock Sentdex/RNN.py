# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:55:38 2017

@author: Jonathan Mak
"""
import os
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
import time

tf.reset_default_graph()

training_rate = 0.001

max_step = 18
n_lex = 13049
n_hidden = 500
n_output = 2
batch_size = 100

x = tf.placeholder('float', [None, max_step, n_lex])
y = tf.placeholder('float', [None, n_output])

weight = tf.Variable(tf.truncated_normal([500, n_output]))
bias = tf.Variable(tf.truncated_normal([n_output]))

def RNN(x, weights = weight, biases = bias):
    x = tf.unstack(x, max_step, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 0.5)
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

start = time.time()

init = tf.global_variables_initializer()
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        train_chunks = [x for x in os.listdir('c:/users/samak/desktop/words/train')]
        test_chunks = [x for x in os.listdir('c:/users/samak/desktop/words/test')]
        train_y = pickle.load(open('./pickle/train_y.p', 'rb'))
        test_y = pickle.load(open('./pickle/test_y.p', 'rb'))
        for epoch in range(10):
            train_y_index = 0
            test_y_index = 0
            train_acc = []
            test_acc = []
            
            for i in range(len(train_chunks)):
                batch_num = int((train_y_index + batch_size)/batch_size)
                batch_x = pickle.load(open('c:/users/samak/desktop/words/train/train_x' + str(batch_num) + '.p', 'rb'))
                batch_y = train_y[train_y_index : train_y_index + batch_size]
                sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
                acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
                train_acc.append(acc)
                train_y_index += batch_size
            print('epoch ', str(epoch+1) + ' training accuracy: ', sum(train_acc)/len(train_acc))
            
            for i in range(len(test_chunks)): 
                batch_num = int((test_y_index + batch_size)/batch_size)
                batch_x = pickle.load(open('c:/users/samak/desktop/words/test/test_x' + str(batch_num) + '.p', 'rb'))
                batch_y = test_y[test_y_index : test_y_index + batch_size]
                acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
                test_acc.append(acc)
                test_y_index += batch_size
            print('epoch ', str(epoch+1) + ' testing accuracy: ', sum(test_acc)/len(test_acc))

end = time.time()
print('Total RNN runtime: ', int(end-start), 's')

#test_x = pickle.load(open('c:/users/samak/desktop/words/chunk1.p', 'rb'))
#with tf.device('/cpu:0'):
#    with tf.Session() as sess:
#        sess.run(init)
#        test = sess.run(prediction, feed_dict={x:test_x})
        