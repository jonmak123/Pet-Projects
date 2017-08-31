# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:37:20 2017

@author: jmak
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


tf.reset_default_graph()

##### LOAD ONLY IF NOT HAVING BEEN LOADED #####
if 'mnist' not in vars():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    
##### LOAD IMAGE ARRAYS AND LABELS #####
img_train = mnist.train.images
img_test = mnist.test.images

lbl_train = mnist.train.labels
lbl_test = mnist.test.labels


##### TO SHOW THE NUMBER #####
def show_img(array_1d):
    plt.imshow(array_1d.reshape((28, 28)))
    
    
##### DEFINE PARAMETERS #####
n_height = 28
n_width = 28
conv1_out_chnl = 10
conv1_window_size = 3
n_output = 10
batch_size = 250
dense1_size = 250
pool_filter_size = 2


##### CREATE TF VARIABLE #####
f_conv1 = tf.Variable(tf.truncated_normal([conv1_window_size, conv1_window_size, 1, conv1_out_chnl], 0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [conv1_out_chnl]))
w_dense1 = tf.Variable(tf.truncated_normal(shape = [int(n_height * n_width / (pool_filter_size ** 2) * conv1_out_chnl), dense1_size], stddev = 0.1))
b_dense1 = tf.Variable(tf.constant(0.1, shape = [dense1_size]))
keep_prob = tf.placeholder(tf.float32)
w_output = tf.Variable(tf.truncated_normal(shape = [dense1_size, 10], stddev = 0.1))
b_output = tf.Variable(tf.constant(0.1, shape = [10]))

##### CREATE TF PLACEHOLDER #####
x = tf.placeholder(tf.float32, shape=[None, n_height * n_width])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
    
    
def conv_model(input_layer):
    ## RESHAPE TO BATCH * 28 * 28 * 1
    input_layer = tf.reshape(input_layer, [-1, n_height, n_width, 1])
    ## FIRST CONV LAYER
    conv1 = tf.nn.conv2d(input_layer, f_conv1, [1, 1, 1, 1], 'SAME')
    conv1 = tf.nn.relu(conv1 + b_conv1)
    ## MAX POOLING
    pool1 = tf.nn.max_pool(conv1, [1, pool_filter_size, pool_filter_size, 1], [1, pool_filter_size, pool_filter_size, 1], 'SAME')
    ## DENSELY CONNECTED LAYER
    dense1 = tf.matmul(tf.reshape(pool1, [-1, int(n_height * n_width / (2 ** 2) * conv1_out_chnl)]), w_dense1)
    dense1 = tf.nn.relu(dense1) + b_dense1
    ## DROUPOUT
    drop1 = tf.nn.dropout(dense1, keep_prob)
    ## OUTPUT LAYER
    output = tf.matmul(drop1, w_output)
    output = tf.nn.softmax(output + b_output)
    return(output)
    
pred_y = conv_model(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred_y))
optimise = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
def train_model(ep = 20):  
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        overall_log = [[], []]
        
        for epoch in range(ep):
            train_acc_log = []
            batch_num = 0   
            
            ##### TRAIN #####
            for i in range(int(len(img_train)/batch_size)):
                train_x = img_train[batch_num : batch_num + batch_size]
                train_y = lbl_train[batch_num : batch_num + batch_size]
                _, acc = sess.run([optimise, accuracy], feed_dict = ({x : train_x, y : train_y, keep_prob : 0.5}))
                train_acc_log.append(acc)
                batch_num += batch_size
            overall_log[0].append(np.mean(train_acc_log))
            
            ##### TEST #####
            test_acc_log = []
            batch_num = 0  
            
            for i in range(int(len(img_test)/batch_size)):
                test_x = img_test[batch_num : batch_num + batch_size]
                test_y = lbl_test[batch_num : batch_num + batch_size]
                acc = sess.run(accuracy, feed_dict = ({x : test_x, y : test_y, keep_prob : 1}))
                test_acc_log.append(acc)
                batch_num += batch_size
            overall_log[1].append(np.mean(test_acc_log))
            
            print('epoch %d, train acc %g; test acc %g' % (epoch + 1, np.mean(train_acc_log), np.mean(test_acc_log)))
        
        ##### PLOT CONVERGENCE #####
        plt.figure()    
        ax1 = plt.plot(range(ep), overall_log[0], color = 'red', label = 'Train')
        ax2 = plt.plot(range(ep), overall_log[1], color = 'green', label = 'Test')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.title('ConvNet Convergence')
        
        saver = tf.train.Saver()
        saver.save(sess)
    
    
if __name__ == '__main__':
    train_model()