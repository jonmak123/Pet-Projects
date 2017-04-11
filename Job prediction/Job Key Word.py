# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 00:15:30 2017

@author: Jonathan Mak
"""

import urllib
import bs4 as bs
import json
import _pickle as pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

all_jobs = ['teacher', 'engineer', 'accountant', 'health', 'ICT', 'lawyer', 'restaurant', 'government', 'driver']
eng = ['mechanical engineer', 'chemical engineer', 'electrical engineer', 'mining engineer', 'civil engineer']

def compile_jobs(search_name):
    d_list = []
    
    loop_next_page = True
    search_name = '-'.join(search_name.split()) + '-jobs'
    url = 'https://www.seek.com.au/' + search_name
    
    ############# Get webpage ############
    while loop_next_page:
        try:
            print(url)
            sauce = urllib.request.urlopen(url)
            soup = bs.BeautifulSoup(sauce, 'html.parser') 
            
            next_page_link = soup.find_all('link', rel="next", href=True)
            jobs = [json.loads(x.text) for x in soup.find_all("script",type="application/ld+json")]
            d_list +=jobs
        
            if len(next_page_link) == 0:
                loop_next_page = False
                print('Scraping completed')
            else:
                url_last = url
                url = next_page_link[0]['href']
        except Exception:
            dummy_input = input('Input anything to continue:')
            url = url_last
    ############# Get webpage ############
    
    ############# Get text ############
    for x in d_list:
        try:
            url = x['url']
            print(x['title'], '---', url)
            sauce = urllib.request.urlopen(url)
            soup = bs.BeautifulSoup(sauce, 'html.parser') 
            for paragraph in soup.find_all('div', class_="job-template__wrapper"):
                job_text = paragraph.text
                x['job_text'] = job_text
        except Exception:
            continue
    ############# Get text ############
    
    pickle.dump(d_list, open( search_name + ".p", "wb" ))
#    return(d_list)


def word_count_1(d): #count 1 if exists in job
    word_dict=dict()
    
    words = []
    for job in d:
        text = job.get('job_text')
        if text == None:
            text = ""
        text = text.strip()
        str_to_replace = ['\n', '\r', '.', ',', '(', ')', '/', ':', '|', '&', "'", ';', '"', '?', '#']
        for x in str_to_replace:
            text = text.replace(x, ' ')
        text = ' '.join(text.split())
        job['clean_text'] = text
    
    words = [j.get('clean_text') for j in d]
    words_list = list(set([s.lower() for s in (' '.join(words)).split()]))
    
    word_dict=dict()
    for word in words_list:
        counter = 0
        for job in words:
            if word in job:
                counter+=1
        word_dict[word] = counter
    
    df = pd.DataFrame.from_dict(word_dict, orient='index')
    
    return(df)

def word_count_2(d): #count total in job
    word_dict=dict()
    
    words = []
    for job in d:
        text = job.get('job_text')
        if text == None:
            text = ""
        text = text.strip()
        str_to_replace = ['\n', '\r', '.', ',', '(', ')', '/', ':', '|', '&', "'", ';', '"', '?', '#']
        for x in str_to_replace:
            text = text.replace(x, ' ')
        text = ' '.join(text.split())
        job['clean_text'] = text
    
    words = [j.get('clean_text') for j in d]
    words_all = [s.lower() for s in (' '.join(words)).split()]
    words_list = list(set([s.lower() for s in (' '.join(words)).split()]))
    
    word_dict=dict()
    for word in words_list:
        counter = words_all.count(word)
        word_dict[word] = counter
    
    df = pd.DataFrame.from_dict(word_dict, orient='index')
    df.columns = ['count']
    
    return(df)

def word_cloud(df):
    list_of_keywords = ['excel', 'vba', 'python', 'r', 'sas', 'analytics', 'technology', 'microsoft', 'database', 'data', 'decision', 'strategy', 'reporting', 'oracle', 'cloud', 'solution', 'communication', 'written', 'verbal']

    df_kw = df.loc[df.index.isin(keywords), :].sort_values(by='count', ascending=False)
    txt_dict = df_kw.to_dict()
    wordcloud = WordCloud(collocations = False).generate_from_frequencies(txt_dict['count'])

#    full_text = []
#    for keyword in keywords:
#        string = [str(keyword)] * df[df.index==keyword]['count'][0]
#        full_text = full_text + string
#    full_text = ' '.join(full_text)
#    wordcloud = WordCloud(collocations = False).generate(full_text)
    
    f1, ax = plt.subplots(2, sharex=False, sharey=False)
    ax[0].imshow(wordcloud, interpolation='bilinear')
    ax[0].axis("off")
    sns.barplot(df_kw.index, df_kw['count'], ax=ax[1])
    plt.xticks(rotation=45)
    ax[0].set_title('Word Count')

def make_lexicon(d_of_ds):
    lexicon = []
    for d in d_of_ds:
        for job in d_of_ds.get(d):
            text = job.get('job_text')
            if text == None:
                text = ""
            text = text.strip()
            str_to_replace = ['\n', '\r', '.', ',', '(', ')', '/', ':', '|', '&', "'", ';', '"', '?', '#']
            for x in str_to_replace:
                text = text.replace(x, ' ')
            lex = text.split()
            job['text_list'] = lex
            lexicon += lex
    lexicon = list(set(lexicon))
    return(lexicon)

def count_words(d_of_ds, lexicon):
    for d in d_of_ds:
        for job in d_of_ds.get(d):
            text = job.get('text_list')
            word_vector = np.zeros(len(lexicon))
            for word in text:
                word_vector[lexicon.index(word)] += 1
            job['lex_count'] = word_vector
    
def prepare_train_set1(d_of_ds, output_list):
    xy = []
    for d in d_of_ds:
        classifier = np.zeros(len(output_list))
        classifier[output_list.index(d)] = 1
        for job in d_of_ds.get(d):
            lex = job.get('lex_count')
            xy.append([lex, classifier])
    return(xy)
    
def prepare_train_set2(d_of_ds, output_list, lexicon):
    for d in d_of_ds:
        xy = []
        classifier = np.zeros(len(output_list))
        classifier[output_list.index(d)] = 1
        for job in d_of_ds.get(d):
            print(job['title'])
            word_vector = np.zeros(len(lexicon))
            unique = list(set(job.get('text_list')))
            for word in unique:
                word_vector[lexicon.index(word)] += 1
            xy.append([word_vector, classifier])
        pickle.dump(xy, open('xy_' + d + '.p', 'wb'))
#    return(xy)
    
    
    
    
    
    