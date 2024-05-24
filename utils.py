import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sklearn

import re
import contractions
import unicodedata
from bs4 import BeautifulSoup
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopword = stopwords.words('english')
  
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()




def processing(text):
    text = json.loads(text)
    return text

def title_fn(dic):
    text = dic.get('title')
    if text != None:
        return text
    else:
        return "unknown_title"
        
def body_fn(dic):
    text = dic.get('body')
    if text != None:
        return text
    else:
        return "unknown_body"
    
def url_fn(dic):
    text = dic.get('url', 'unknown_url')
    if text != None:
        return text
    else:
        return "unknown_url"
    
def transform_data(new_data):
    new_data.boilerplate = new_data.boilerplate.apply(lambda text: processing(text))
    new_data['title'] = new_data.boilerplate.apply(title_fn)
    new_data['Body'] = new_data.boilerplate.apply(body_fn)
    new_data['b_url'] = new_data.boilerplate.apply(url_fn)

    return new_data


def text_preprocess(text):
    
    try:
        contractions.fix(text)
    except:
        text = text
    else:
        text = contractions.fix(text)
    finally:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')## Removing/normalising accented characters.
        text = re.sub(r' @[^\s]*',"",text)#Remove @elements
        #text = re.sub(r'RT[^A-Za-z]+',"",text)#Remove RT RETWEET tag
        text = re.sub(r'(([A-Za-z0-9._-]+)@([A-Za-z0-9._-]+)(\.)([A-Za-z]{2,8}))',"",text) #email
        text = re.sub(r'([A-Za-z0-9]+)(\*)+([A-Za-z0-9]+)','starword',text)# replacing ***words with "star_word"
        text = re.sub(r'((https|http|ftp)?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'," ",text) #urls
        text = BeautifulSoup(text, 'lxml').get_text(" ")#tag removal
        text = text.lower() #Lowering the characters
        #text = re.sub(['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'],'',text)
        text =  re.sub(r'[^\w\s]', '', text)
        text =  re.sub(r'[0-9]', '', text)
        tokens = word_tokenize(text)
        #text = [ps.stem(i) for i in tokens if i not in stopword]
        text = [lemmatizer.lemmatize(i) for i in tokens if i not in stopword]
        text = " ".join(text)
        
    return text




