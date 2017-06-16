# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:24:56 2016

@author: J-Mac
"""
import pandas as pd
from pandas_datareader import data as web
import pickle

start = '5-30-2016'
end = '5-29-2017'
stock = ['AAPL', 'GOOG', 'IBM', 'NFLX', 'MSFT', 'ORCL', 'TSLA']
#stock = ['AAPL']
price = pd.DataFrame()
for s in stock:
    data = web.DataReader(s, 'google', start, end)
    data['Stock'] = s
    price = price.append(data)
pickle.dump(price, open('price.p', 'wb'))

