n# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 09:38:48 2017

@author: Jonathan Mak
"""

import urllib
import pandas as pd
import numpy as np
import requests
import json

listingType = 'Share'
pagesize = 10

def get_agent_info():

    url_listing_search = 'https://api.domain.com.au/v1/listings/_search'
    url_listing_id = 'https://api.domain.com.au/v1/listings/'
    url_agent = 'https://api.domain.com.au/v1/agents/'
    
    headers = {'Authorization' : 'Bearer 4uwj4x9qe24jf5s5szruuahx', 
               'X-Originating-Ip' : '103.95.194.16',
               'Content-Type': 'application/json'}
    
    body = {'listingType' : listingType,
            'pagesize' : pagesize}
    body = json.dumps(body)
    
    r_listing_search = requests.post(url_listing_search,
                                     data=body,
                                     headers=headers)
    
    r_listing_search = json.loads(r_listing_search.content)
    
    for i in range(len(r_listing_search)):
        listing_id = r_listing_search[i]['listing']['id']
        r_listing_id = requests.get(url_listing_id + str(listing_id), 
                                    headers=headers)
        r_listing_id = json.loads(r_listing_id.content)
        agent_id = r_listing_id['advertiserIdentifiers']['contactIds']
        
        r_listing_search[i]['listing']['agent_INFO'] = dict()
        for j in range(len(agent_id)):        
            r_agent = requests.get(url_agent + str(agent_id[j]), 
                                   headers=headers)
            r_agent = json.loads(r_agent.content)
            r_listing_search[i]['listing']['agent_INFO'][str(j)] = r_agent
    
    return(r_listing_search)

 

