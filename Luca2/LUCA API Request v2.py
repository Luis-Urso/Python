# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:49:36 2022
last Update on Apr 18th 

@author: Luis A. Urso
"""

import requests
import json


def lookup_qna(topic,question):
    
    # Define Topic list and the associated QnA KB Keys
    
    qna_kbs =['None',
              'Greeting',
              'Password',
              'Travel',
              'Data Integrity',
              'UNeRecruit']
    
    qna_kb_keys=['c68dc93f-e17c-4aeb-ab73-6d76d67162b0',
                 'c68dc93f-e17c-4aeb-ab73-6d76d67162b0',
                 '51c49955-5944-44f6-b8cb-ed73ef48080d',
                 '222519b4-7250-46b3-9594-dd312f88e912',
                 'e97c6ff6-7b72-4732-af35-ead91f633e5c',
                 'b550b632-691f-48dc-8494-bae709673beb']
    
    qna_url_key = ''
    
    tcount=0
    
    for i in qna_kbs:     
   
        if(topic==i):
            qna_url_key=qna_kb_keys[tcount]
            break
        else: 
            tcount = tcount+1
              
    qna_url="https://z1-poc-luca-qna-app.azurewebsites.net/qnamaker/knowledgebases/"+qna_url_key+"/generateAnswer"
    
    qna_headers = {
        'Authorization':'EndpointKey 7f617c7c-e96d-4f13-bbf3-568ad07a38d7',
        'Content-type': 'application/json',
    }
    
    qna_query="{'question':'"+question+"'}"
    #print(qna_query)
    
    qna_resp = requests.post(qna_url,headers=qna_headers,data=qna_query)
     
    data=qna_resp.json()  
    
    print(float(data['answers'][0]['score']))
    
    if (qna_resp.status_code == 200):
        
       if (float(data['answers'][0]['score']))>=50:
           return(data['answers'][0]['answer'])
       else:
           return("Could not find an appropriate answer for your question")
    else:
       return("Error:"+str(qna_resp.status_code))
             



## Use LUIS to predict the topics to consider to look into
## QnA Maker

def ask_luca(question):

    luis_url="https://z1-poc-luca-cog-services.cognitiveservices.azure.com/luis/prediction/v3.0/apps/c0b33e5f-4b54-4537-9ee7-a09af0876c6e/slots/production/predict?verbose=true&show-all-intents=true&log=true&subscription-key=521967aab5ab4ae9b9d9632473e7a92f&query="
    
    luis_query=question
    
    luis_parm=luis_url+luis_query
    
    luis_resp = requests.get(luis_parm)
    
    data=luis_resp.json()
    
    top_intent=data['prediction']['topIntent']
    
    print(data)
        
    if (luis_resp.status_code == 200):
    
        print("Luis: Chasing Intent...")
        data=data['prediction']['intents']

        tags=[]
        responses=[]
        
        for tag in data:
            tags.append(tag)
                       
        for i in range(len(tags)):             
            score = str(data[tags[i]])
            prob = float(score[10:16])
            
            ## Call QnA Maker topics those are higher or equal to defined
            ## threshold below. 
            
            print(score)
            
            if prob >= 0.50:
                
                print(tags[i]) #+ ': ' + # str(data[tags[i]])) 
                
                ## Look for QnA Responses
                
                responses.append(lookup_qna(tags[i],question))
                
            else:
                
                print("Luis: Using Top Intent")
                
                responses.append(lookup_qna(top_intent,question))
                
                ## Luis return was None
                #responses.append(lookup_qna("None",question))
        
    else:
        responses[0]=("Error:"+luis_resp.status_code)
    
    return(responses)



#### 
#### 

print("LLLL           UUUU    UUUU   CCCCCCCCCC        AAAAA")
print("LLLL           UUUU    UUUU   CCCCCCCCCC      AAAA AAAA")
print("LLLL           UUUU    UUUU   CCCC           AAAA   AAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAA     AAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAAAAAAAAAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAAAAAAAAAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAA     AAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAA     AAAA")
print("LLLL           UUUU    UUUU   CCCC          AAAA     AAAA")
print("LLLLLLLLLLL    UUUUUUUUUUUU   CCCCCCCCCCC   AAAA     AAAA")
print("LLLLLLLLLLL    UUUUUUUUUUUU   CCCCCCCCCCC   AAAA     AAAA")
print("")
print("Luca is back ! Version 2.0 or Lucca power 2")
print("")
print("")
print("How can I help you today?")

qst=''

while qst!="bye":
    
    qst = input("You: ")
    if qst=="bye": break
    rsp = ask_luca("Luca says: "+str(qst))
        
    print(rsp)
    print('')


    
    
