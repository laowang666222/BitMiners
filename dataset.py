import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sklearn
from sklearn.model_selection import train_test_split
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
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

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
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8',
                                                                                    'ignore')  ## Removing/normalising accented characters.
        text = re.sub(r' @[^\s]*', "", text)  # Remove @elements
        # text = re.sub(r'RT[^A-Za-z]+',"",text)#Remove RT RETWEET tag
        text = re.sub(r'(([A-Za-z0-9._-]+)@([A-Za-z0-9._-]+)(\.)([A-Za-z]{2,8}))', "", text)  # email
        text = re.sub(r'([A-Za-z0-9]+)(\*)+([A-Za-z0-9]+)', 'starword', text)  # replacing ***words with "star_word"
        text = re.sub(
            r'((https|http|ftp)?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
            " ", text)  # urls
        text = BeautifulSoup(text, 'lxml').get_text(" ")  # tag removal
        text = text.lower()  # Lowering the characters
        # text = re.sub(['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'],'',text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[0-9]', '', text)
        tokens = word_tokenize(text)
        # text = [ps.stem(i) for i in tokens if i not in stopword]
        text = [lemmatizer.lemmatize(i) for i in tokens if i not in stopword]
        text = " ".join(text)

    return text

class TextDataSet(Dataset):
    def __init__(self, filepath,max_len=320):
        self.max_len = max_len
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
        data = pd.read_csv(filepath, sep='\t')
        new_data = transform_data(data)
        new_data['full'] = new_data['title'] + new_data['Body'] + new_data['b_url']
        new_data['full'] = new_data['full'].apply(lambda x: text_preprocess(x))
        self.rows=new_data.full.tolist()
        self.labels = new_data.label.values
        # print(type(self.rows))
        # print(self.rows)


    def __getitem__(self, index):
        tokenized = self.tokenizer.tokenize(self.rows[index])
        tokenized_with = ["[CLS]"] + tokenized  # 添加[CLS]标志
        tok_ids = self.tokenizer.convert_tokens_to_ids(tokenized_with)  # 转换为id
        length = len(tok_ids)
        # padding
        if len(tok_ids) < self.max_len:
            tok_ids += [0] * (self.max_len - len(tok_ids))
        else:
            tok_ids = tok_ids[:self.max_len]
        tok_tensor = torch.tensor(tok_ids)

        #读取标签


        return tok_tensor,self.labels[index]

    def __len__(self):
        return len(self.rows)


if __name__=="__main__":
    trainset= TextDataSet('train.tsv')
    print(trainset[0])
