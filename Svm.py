import numpy as np
from scipy.spatial import distance
from collections import Counter

from sklearn.model_selection import train_test_split
from utils import transform_data, text_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC



def get_dataset():
    train_path="train.tsv"
    data=pd.read_csv(train_path,sep='\t')
    new_data = transform_data(data)
    new_data['full'] = new_data['title'] + new_data['Body'] + new_data['b_url']
    new_data['full'] = new_data['full'].apply(lambda x: text_preprocess(x))
    cv_tf = TfidfVectorizer()
    transformed_data = cv_tf.fit_transform(new_data.full)
    X = transformed_data
    y = new_data.label.values
    n_components = 100  

    svd = TruncatedSVD(n_components)
    X_reduced = svd.fit_transform(X)
    
    return X_reduced,y

if __name__ == '__main__':
    X,y=get_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC(random_state=42)
    svm.fit(X_train, y_train)
    svm_score = svm.score(X_val, y_val)
    print(f"SVM score: {svm_score}")
