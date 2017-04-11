# -*- coding: utf-8 -*-
import tensorflow as tf
import _pickle as pickle
import numpy as np
from scipy.stats import rankdata
import random
import matplotlib.pyplot as plt

train_eng = pickle.load(open('train_eng.p', 'rb'))
lexicon = pickle.load(open('lexicon_eng.p', 'rb'))

eng = ['mechanical engineer', 'chemical engineer', 'electrical engineer', 'mining engineer', 'civil engineer']

def divide_sets(train_set, test_size = 0.1):
    random.shuffle(train_eng)
    xx = [a[0] for a in train_eng]
    yy = [a[1] for a in train_eng]
    
    testing_size = int(test_size*len(train_eng))
       
    train_x = list(xx[:-testing_size])
    train_y = list(yy[:-testing_size])
    test_x = list(xx[-testing_size:])
    test_y = list(yy[-testing_size:])
    
    return(train_x,train_y,test_x,test_y)

train_x,train_y,test_x,test_y = divide_sets(train_eng)

n_nodes_hl1 = 1000
n_nodes_hl2 = 500
n_nodes_hl3 = 100

n_classes = int(len(train_y[0]))
batch_size = 100
hm_epochs = 5


tf.reset_default_graph()
x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

#hidden_3_layer = {'f_fum':n_nodes_hl3,
#                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

saver=tf.train.Saver()

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
#    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
#    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l2,output_layer['weight']), output_layer['bias'])
    return(output)

def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    epoch_list = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
	    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            
            epoch_list.append(epoch_loss)
            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        
        saver.save(sess, './temp/model.ckpt')
        
        f, ax = plt.subplots()
        ax.plot(range(len(epoch_list)), epoch_list)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

        return(sess)
    
#train_neural_network(x,y)

def use_neural_network(adj_string_with_space):
#    tf.reset_default_graph()
    
    word_vector=np.zeros(len(lexicon))
    words = adj_string_with_space.split()
    for word in words:
        if word in lexicon:
            word_vector[lexicon.index(word)] += 1

    sess = tf.InteractiveSession()
    prediction = neural_network_model(x)
    sess.run(tf.global_variables_initializer())
    tf.train.import_meta_graph('./temp/model.ckpt.meta')
    saver.restore(sess,"./temp/model.ckpt")
    result = sess.run(prediction, feed_dict={x:[word_vector]})
    result = list(zip(eng, list(result[0])))
    return(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    