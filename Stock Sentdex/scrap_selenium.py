# -*- coding: utf-8 -*-
"""
Created on Sun May 28 20:07:31 2017

@author: Jonathan Mak
"""

from selenium import webdriver
import datetime
import time
import bs4 as bs
import pickle
import pandas as pd
import numpy as np

stock = ['AAPL', 'GOOG', 'IBM', 'NFLX', 'MSFT', 'ORCL', 'TSLA']

def use_selenium(s, y, m, d, list_to_add):
    for num in range(0, 20, 10):
        print(d, m, y, sep='/')
        url = 'https://www.google.com.au/search?q='+s+'&tbs=cdr%3A1%2Ccd_min%3A'+str(m)+'%2F'+str(d)+'%2F'+str(y)+'%2Ccd_max%3A'+str(m)+'%2F'+str(d)+'%2F'+str(y)+'&tbm=nws&start='+str(num)
        driver.get(url)
        time.sleep(3)
        content = driver.page_source
        soup = bs.BeautifulSoup(content, 'html.parser')
        head = soup.find_all('a', {'class':'l _PMs'})
        for h in head:
            entry = (datetime.datetime(y, m, d), s, h.get_text(' ', strip=True))
            list_to_add.append(entry)
    return(list_to_add)

driver = webdriver.Firefox()
data = []
num_days = 3
date_list = [datetime.datetime.today()-datetime.timedelta(days=x) for x in range (1, num_days+1)]
for s in stock:
    for date in date_list:
        y = date.year
        m = date.month
        d = date.day
        data = use_selenium(s, y, m, d, data)
pickle.dump(data, open('data.p', 'wb'))
        
######## Read in dataframe #########
df = pd.DataFrame.from_records(data, columns=['date', 'stock', 'header'])












