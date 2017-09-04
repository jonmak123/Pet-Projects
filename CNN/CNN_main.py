# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:37:20 2017

@author: jmak
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

##### FLUSH ANY PRE_EXISTING GRAPH #####
tf.reset_default_graph()


##### LOAD ONLY IF NOT HAVING BEEN LOADED #####
if 'mnist' not in vars():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        
    
##### LOAD IMAGE ARRAYS AND LABELS #####
img_train = mnist.train.images
img_test = mnist.test.images

lbl_train = mnist.train.labels
lbl_test = mnist.test.labels


##### (UTILS) TO SHOW THE NUMBER #####
def show_img(array_1d):
    plt.imshow(array_1d.reshape((28, 28)))
    
    
##### DEFINE PARAMETERS #####
n_height = 28
n_width = 28
conv1_out_chnl = 16
conv2_out_chnl = 10
conv1_window_size = 3
conv2_window_size = 2
n_output = 10
batch_size = 250
dense1_size = 250
pool1_filter_size = 2
pool2_filter_size = 2
pool1_stride = 2
pool2_stride = 1


##### DEFINE WHICH DEVICE TO WORK WITH #####
with tf.device('/cpu:0'):
   
    ##### SUMMARY OPS WRAPPER FUNC FOR TENSORBOARD ##### 
    def variable_summaries(var):
        ## TENSORBOARD SUMMARY ONLY WORKS WITH CPU IT SEEMS
        with tf.device('/cpu:0'):
            ## THESE ARE FOR WEIGHTS
            with tf.name_scope('summaries'):
                tf.summary.scalar('mean', tf.reduce_mean(var))
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
    
    ##### CREATE TF PLACEHOLDER #####
    x = tf.placeholder(tf.float32, shape=[None, n_height * n_width])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)     
        
    def conv_model(input_layer):
        
        ## FIRST CONV LAYER
        with tf.name_scope('Conv_Layer_1'):
            ## RESHAPE TO BATCH * 28 * 28 * 1
            input_layer = tf.reshape(input_layer, [-1, n_height, n_width, 1])
            f_conv1 = tf.Variable(tf.truncated_normal([conv1_window_size, conv1_window_size, 1, conv1_out_chnl], 0.1), name = 'conv1_kernel')
            b_conv1 = tf.Variable(tf.constant(0.1, shape = [conv1_out_chnl]), name = 'conv1_bias')
            conv1 = tf.nn.conv2d(input_layer, f_conv1, [1, 1, 1, 1], 'SAME', name = 'conv1_main')
            conv1 = tf.nn.relu(conv1 + b_conv1, name = 'conv1_actn')
            ## MAX POOLING 1
            pool1 = tf.nn.max_pool(conv1, [1, pool1_filter_size, pool1_filter_size, 1], [1, pool1_stride, pool1_stride, 1], 'SAME', name = 'maxpool1')
            ## WRITE SUMMARY
            variable_summaries(conv1)
            variable_summaries(pool1)
            variable_summaries(f_conv1)            
            tf.summary.image('input', input_layer, 1)
            tf.summary.image('conv1', tf.reshape(tf.transpose(conv1, [0, 3, 1, 2]), [-1, 28, 28, 1]), 1)
            tf.summary.image('pool1', tf.reshape(tf.transpose(conv1, [0, 3, 1, 2]), [-1, 14, 14, 1]), 1)
        
        ## SECOND CONV LAYER
        with tf.name_scope('Conv_Layer_2'):
            f_conv2 = tf.Variable(tf.truncated_normal([conv2_window_size, conv2_window_size, 16, conv2_out_chnl], 0.1), name = 'conv2_kernel')
            b_conv2 = tf.Variable(tf.constant(0.1, shape = [conv2_out_chnl]), name = 'conv2_bias')
            conv2 = tf.nn.conv2d(pool1, f_conv2, [1, 1, 1, 1], 'SAME', name = 'conv2_main')
            conv2 = tf.nn.relu(conv2 + b_conv2, name = 'conv2_actn') 
            ## MAX POOLING 2
            pool2 = tf.nn.max_pool(conv2, [1, pool2_filter_size, pool2_filter_size, 1], [1, pool2_stride, pool2_stride, 1], 'SAME', name = 'maxpool2')
            ## WRITE SUMMARY
            variable_summaries(f_conv2)
            variable_summaries(conv2)
            tf.summary.image('conv2', tf.reshape(tf.transpose(conv1, [0, 3, 1, 2]), [-1, 14, 14, 1]), 1) 
            tf.summary.image('pool2', tf.reshape(tf.transpose(conv1, [0, 3, 1, 2]), [-1, 14, 14, 1]), 1) 
            
        with tf.name_scope('Dense_Layer'):
            ## DENSELY CONNECTED LAYER
            w_dense1 = tf.Variable(tf.truncated_normal(shape = [int(n_height * n_width / (pool1_stride ** 2) * conv2_out_chnl), dense1_size], stddev = 0.1), name = 'dense_weight')
            b_dense1 = tf.Variable(tf.constant(0.1, shape = [dense1_size]), name = 'dense_bias')
            dense1 = tf.matmul(tf.reshape(pool2, [-1, 1960]), w_dense1, name = 'dense_layer_main')
            dense1 = tf.nn.relu(dense1 + b_dense1, name = 'dense_layer_actn')  
            ## DROUPOUT
            drop1 = tf.nn.dropout(dense1, keep_prob, name = 'dropout')
            ## WRITE SUMMARY
            variable_summaries(w_dense1)
            variable_summaries(dense1)
            
        with tf.name_scope('Output_Layer'):
            ## OUTPUT LAYER
            w_output = tf.Variable(tf.truncated_normal(shape = [dense1_size, 10], stddev = 0.1), name = 'output_weight')
            b_output = tf.Variable(tf.constant(0.1, shape = [10]), name = 'output_bias')
            output = tf.matmul(drop1, w_output, name = 'output_main')
            output = tf.nn.softmax(output + b_output, name = 'output_softmax')
            
        return(output)
        
        
    ##### COMBINE LAYERS INTO MODEL #####    
    pred_y = conv_model(x)
    
    
    ##### DEFINE OPS TO CALCULATE ACC #####
    with tf.name_scope('Accuracy'):  
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred_y))
        correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        
        
    ##### DEFINE OPTIMISATION ALGO #####    
    with tf.name_scope('Optimise'):
        optimise = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    
    merged = tf.summary.merge_all()


def train_model(ep = 20, plot_converg = True):  
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        overall_log = [[], []]
        
        with tf.device('/cpu:0'):  
            writer = tf.summary.FileWriter('./model/events/1')
            writer.add_graph(sess.graph)
        
        for epoch in range(ep):
            
            ##### TRAIN #####
            train_acc_log = []
            batch_num = 0              
            for i in range(int(len(img_train)/batch_size)):
                train_x = img_train[batch_num : batch_num + batch_size]
                train_y = lbl_train[batch_num : batch_num + batch_size]              
                if i % 10 ==0:
                    ## RUN SUMMARY WRITER EVERY 10 STEPS
                    summ = sess.run(merged, feed_dict = ({x : train_x, y : train_y, keep_prob : 0.5}))                
                    writer.add_summary(summ, epoch * int(len(img_train)/batch_size) + i)
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
        if plot_converg is True:
            plt.figure()    
            plt.plot(range(ep), overall_log[0], color = 'red', label = 'Train')
            plt.plot(range(ep), overall_log[1], color = 'green', label = 'Test')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
            plt.xlabel('Training Epoch')
            plt.ylabel('Accuracy')
            plt.title('ConvNet Convergence')
        
        ##### SAVE MODEL FOR REPEATED USE #####
        with tf.name_scope('Save_model'):
            saver = tf.train.Saver()
            saver.save(sess, './model/mymodel')     
                  
    return(sess)
    
    
if __name__ == '__main__':
    sess1 = train_model(ep = 20)