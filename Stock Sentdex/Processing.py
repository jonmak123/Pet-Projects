# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:24:56 2016

@author: J-Mac
"""

import datetime as dt
import numpy as np
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
import string

###### Open files ######
scraped = pickle.load(open('./Scraping/TSLA.p', 'rb'))
news = pd.DataFrame.from_records(scraped, columns=['Date', 'Stock', 'header'])
#price2 = pickle.load(open('price.p', 'rb'))
price = pd.read_pickle('./pickle/price.p')
price = price.reset_index()

###### pd merge ######
df = pd.merge(news, price, how='outer', on=['Date', 'Stock'])
df['weekday'] = df['Date'].dt.weekday
df = df.loc[df['Open'].isnull()==False, :]
df['perc_ch'] = (df['Close'] - df['Open'])/df['Open']
df['response'] = [1 if x >=0 else 0 for x in df['perc_ch']]

###### tokenise words ######
tokenizer = nltk.tokenize.RegexpTokenizer(r'(?!\d)+\w+')
df['header_token'] = df['header'].apply(lambda x:tokenizer.tokenize(str.lower(str(x))), 1)

###### Check if string is English because I screwed up the scraping ######
def isEnglish(s):
    try:
        str(s)[1].encode('ascii')
    except:
        return False
    else:
        return True

df['ifEng'] = df['header'].apply(isEnglish)
df = df.loc[df['ifEng']==True, :]

###### Shuffle df ######
df = df.sample(frac = 1)

###### make bag of words and create word vector ######
lexicon = list(set([w for l in df['header_token'].tolist() for w in l]))
lines = df['header_token'].tolist()
train_x = lines[:int(len(lines)*0.9)]
test_x = lines[int(len(lines)*0.9):]
max_words = max([len(x) for x in lines])

def prepare_set(data_set, name):
    chunk = []
    counter = 1
    for line in data_set:
        print(counter)
        sentence= []
    #    sentence = np.zeros((max_words, len(lexicon)))
        for word in line: 
    #        vec = np.zeros(len(lexicon))
            vec = [0] * len(lexicon)
            vec[lexicon.index(str(word))] += 1
            sentence.append(vec)
        if len(sentence) < max_words:
            sentence += [[0] * len(lexicon)]*(max_words - len(sentence))
    #        sentence[len(sentence):max_words] = np.zeros(len(lexicon))
        chunk.append(sentence)
        if counter % 100 == 0 and counter != 0 :
            pickle.dump(np.array(chunk, 'float32'), open('c:/users/samak/desktop/words/' + name + str(int(counter/100))+'.p', 'wb'))
            chunk = []
        counter += 1

prepare_set(train_x, 'train_x')
prepare_set(test_x, 'test_x')

y = df['response'].apply(lambda x: [1, 0] if x==0 else [0, 1]).tolist()
train_y = np.array(y[:int(len(lines)*0.9)], 'float32')
test_y = np.array(y[int(len(lines)*0.9):], 'float32')
pickle.dump(train_y, open('./pickle/train_y.p', 'wb'))
pickle.dump(test_y, open('./pickle/test_y.p', 'wb'))





