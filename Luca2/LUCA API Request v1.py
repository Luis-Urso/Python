# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:49:36 2022

@author: WB02554
"""

import requests
import json


## Use LUIS to predict the topics to consider to look into
## QnA Maker

luis_url="https://z1-poc-luca-cog-services.cognitiveservices.azure.com/luis/prediction/v3.0/apps/c0b33e5f-4b54-4537-9ee7-a09af0876c6e/slots/production/predict?verbose=true&show-all-intents=true&log=true&subscription-key=521967aab5ab4ae9b9d9632473e7a92f&query="

luis_query="Find a Job"

luis_parm=luis_url+luis_query

luis_resp = requests.get(luis_parm)

data=luis_resp.json()

print(data)

if (luis_resp.status_code == 200):

    data=data['prediction']['intents']
    
    tags=[]
    
    for tag in data:
        tags.append(tag)
                   
    for i in range(len(tags)):             
        score = str(data[tags[i]])
        prob = float(score[10:16])
        
        ## Call QnA Maker topics those are higher or equal to defined
        ## threshold below. 
        
        if prob >= 0.5:
            print(tags[i] + ': ' + str(data[tags[i]]))                  
    
else:
    print("Error:"+luis_resp.status_code)
    
    
    
    
    
    
