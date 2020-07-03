# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:41:41 2020

@author: pritt
"""
import nltk
import pandas as pd

dataset = pd.read_csv('data.csv')

content = dataset['content']

#Tokenization
token_list=[]
s=""
from nltk.tokenize import sent_tokenize, word_tokenize
for i in content:
    print(i)
    print("--------------------------------------------------------------")
    token_list.append(sent_tokenize(i))
    s=s+ str(sent_tokenize(i))
    

flatList = []
for elem in token_list:
    flatList.extend(elem)
print('Flat List : ', flatList)
    


#word tokenize
wt=[]
for i in flatList:
    wt.append(word_tokenize(i))



flatList2 = []
for elem in wt:
    flatList2.extend(elem)
print('Flat List : ', flatList2)




#POS tagging

tags = nltk.pos_tag(flatList2)


#StopWords removal
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
filtered_sentence = [] 
  
for w in flatList2: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
    
#Stemming
from nltk.stem import PorterStemmer 

ps = PorterStemmer()

for w in filtered_sentence: 
    print(w, " : ", ps.stem(w)) 
    
    
    