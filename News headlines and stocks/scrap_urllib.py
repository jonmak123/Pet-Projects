# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:58:28 2017

@author: Jonathan Mak
"""

import urllib
import bs4 as bs
import requests
import datetime
import pandas as pd
import numpy as np
import pickle

stock = ['AAPL', 'GOOG', 'IBM', 'NFLX', 'MSFT', 'ORCL', 'TSLA']

######## scrape google news #########
def prepare_data(s, y, m, d, list_to_add):
    for num in range(0, 30, 10): #scrape 2 pages of search results
        url = 'https://www.google.com.au/search?q='+s+'&tbs=cdr%3A1%2Ccd_min%3A'+str(m)+'%2F'+str(d)+'%2F'+str(y)+'%2Ccd_max%3A'+str(m)+'%2F'+str(d)+'%2F'+str(y)+'&tbm=nws&start='+str(num)
        r = requests.get(url)
        content = r.text
        soup = bs.BeautifulSoup(content, 'html.parser')
    
        head = soup.find_all('a')
        for h in head:
            if 'class' not in str(h):
                if h.string is not None:
                    if h.string == 'Advanced search':
                        break
                    entry = (datetime.datetime(y, m, d), s, h.string)
                    list_to_add.append(entry)
    return(list_to_add)

######## loop functions over dates #########
num_days = 1
date_list = [datetime.datetime.today()-datetime.timedelta(days=x) for x in range (0, num_days+1)]
for s in stock:
    data = []
    for d in date_list:
        data = prepare_data(s, d.year, d.month, d.day, data)
        print(d.day, d.month, d.year, sep='/')
    pickle.dump(data, open(stock+'.p', 'wb'))

######## Read in dataframe #########
#df = pd.DataFrame.from_records(data, columns=['date', 'stock', 'header'])