# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:01:34 2017

@author: Jonathan Mak
"""

import airbnb
from airbnb import Api
import requests
import json
import datetime
from datetime import datetime as dt


##### GET API SESSION #####
def dev_login(log_name, pw):
    api = Api(log_name, pw)
    return(api)


##### SEARCH GIVEN PARAMETERS, RETURN LISTINGS
def search(guess_number, suburb, city, price_max, price_min):
    url = 'https://api.airbnb.com/v2/search_results?client_id=3092nxybyb0otqw18e8nh5nty&currency=AUD&_format=for_search_results_with_minimal_pricing&_limit=20&_offset=0&fetch_facets=true&guests=' + str(guess_number) + '&ib=false&ib_add_photo_flow=true&location=' + suburb + '%20' + city + '&min_bathrooms=0&min_bedrooms=0&min_beds=1&price_max=' + str(price_max) + '&price_min=' + str(price_min) + '&sort=1'
    r = requests.get(url)
    r = r.content
    r = json.loads(r)
    return(r)


##### GO THROUGH LISTINGS AND FIND SUITABLE PLACES FOR HOMELESS PLACING GIVEN PARAMETERS
def find_places(api, number_days_search, number_days_needed, max_price = 100, min_price = 0):
    search_result = search(1, 'Randwick', 'Sydney', max_price, min_price)
#
    id_needed = []
    for i in range(len(search_result['search_results'])):
        id_ = search_result['search_results'][i]['listing']['id']
        name_ = search_result['search_results'][i]['listing']['name']
        calendar_ = api.get_calendar(id_, calendar_months = 2)
        search_result['search_results'][i]['listing']['calendar'] = calendar_
        search_result['search_results'][0]['listing']['calendar']['calendar_months'][0]['days'][0]
        d = search_result['search_results'][i]['listing']['calendar']
        count = 0
        for k in range(len(d['calendar_months'][0]['days'])):
           date_ = d['calendar_months'][0]['days'][k]['date']
           date_ = dt.strptime(date_, '%Y-%m-%d')
           available_ = d['calendar_months'][0]['days'][k]['available']
#           print(date_)
           if (date_ - dt.now()).days <= number_days_search and (date_ - dt.now()).days >= 0:
               if available_ == True:
#               print(id_, name_, date_, available_)
                   count += 1
        if count >= number_days_needed:
            id_needed.append({'id' : id_, 
                              'name' : name_, 
                              'days available in specified period' : count})
#            print(id_, name_, 'days_available:', count)
    return(id_needed)


if __name__ == '__main__':
    api = dev_login()
#    test = find_places(api, 7, 4)
           