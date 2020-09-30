# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 20:23:19 2020

@author: pritt
"""

import pandas as pd
import os

os.chdir("F:/Pythonfds/NLP_assignments")

data = pd.read_csv('data.csv')

data['word_count'] = data['content'].apply(lambda x: len(str(x).split(" ")))
data[['content','word_count']].head()



data['char_count'] = data['content'].str.len() ## this also includes spaces
data[['content','char_count']].head()

#Average Word Length
#Number of characters(without space count)/Total number of words
def avg_word(sentence):
  words = sentence.split()
  print(words)
  print(len(words))
  print(sum(len(word) for word in words))
  if words == []:
    return 0
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['content'].apply(lambda x: avg_word(x))
data[['content','avg_word']].head()



#Number of special characters
data['hastags'] = data['review'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['review','hastags']].head()

#Number of numerics
data['numerics'] = data['review'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['review','numerics']].head()

#Number of Uppercase words
data['upper'] = data['review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['review','upper']].head()

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
from textblob import TextBlob, Word, Blobber
import nltk
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
                print(ppo, tup)
    except:
        pass
    return cnt

data['noun_count'] = data['content'].apply(lambda x: check_pos_tag(x, 'noun'))
data['verb_count'] = data['content'].apply(lambda x: check_pos_tag(x, 'verb'))
data['adj_count'] = data['content'].apply(lambda x: check_pos_tag(x, 'adj'))
data['adv_count'] = data['content'].apply(lambda x: check_pos_tag(x, 'adv'))
data['pron_count'] = data['content'].apply(lambda x: check_pos_tag(x, 'pron'))
data[['content','noun_count','verb_count','adj_count', 'adv_count', 'pron_count' ]].head()



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv=CountVectorizer()
A_vec = cv.fit_transform(data['content'])
print(A_vec.toarray())

tv=TfidfVectorizer()
t_vec = tv.fit_transform(data['content'])
print(t_vec.toarray())


feature_names = tv.get_feature_names()

dense = t_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

df_c =pd.concat([df,data], axis=1)
df_c.head()