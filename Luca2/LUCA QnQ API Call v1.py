# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:24:51 2022

@author: WB02554
"""


import requests
import json


def ask_qna(question):

    qna_url="https://z1-poc-luca-qna-app.azurewebsites.net/qnamaker/knowledgebases/c68dc93f-e17c-4aeb-ab73-6d76d67162b0/generateAnswer"
    
    qna_headers = {
        'Authorization':'EndpointKey 7f617c7c-e96d-4f13-bbf3-568ad07a38d7',
        'Content-type': 'application/json',
    }
    
    qna_query="{'question':'"+question+"'}"
    #print(qna_query)
    
    qna_resp = requests.post(qna_url,headers=qna_headers,data=qna_query)
     
    data=qna_resp.json()
    
    #print(data)
    
    #print(data['answers'][0]['questions'])
    
    return(data['answers'][0]['answer'])
   # print(data['answers'][0]['score'])
    
   # if (qna_resp.status_code == 200):
    
       # data=data['answers'][0]['answer']
        
       # tags=[]
        
       # for tag in data:
        #    tags.append(tag) 
        
       # print(tags)
    
                   
        #for i in range(len(tags)):             
           # score = str(data[tags[i]])
           # prob = float(score[10:16])
            
            ## Call QnA Maker topics those are higher or equal to defined
            ## threshold below. 
            
            # if prob >= 0.5:
              #  print(tags[i] + ': ' + str(data[tags[i]]))                  
    # else:
       # print("Error:"+qna_resp.status_code)
        
        
        
pergunta = ask_qna("Good Morning")
print(pergunta)