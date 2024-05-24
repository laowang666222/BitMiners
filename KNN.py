import numpy as np
from scipy.spatial import distance
from collections import Counter

from sklearn.model_selection import train_test_split
from utils import transform_data, text_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD


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

class MyKNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = distance.cdist(np.array([test_point]), self.X_train, 'euclidean')[0]
            indices = np.argsort(distances)[:self.n_neighbors]  # 取最近的k个邻居
            neighbors = [self.y_train[i] for i in indices]
            prediction = Counter(neighbors).most_common(1)[0][0]  # 取最多的类别作为预测结果
            predictions.append(prediction)
        return np.array(predictions)

if __name__ == '__main__':
    X,y=get_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = MyKNNClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    print(sum(y_pred==y_val)/len(y_pred))
