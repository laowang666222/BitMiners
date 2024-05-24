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

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class  # 当前节点预测的类别
        self.feature_index = 0  # 用于分裂节点的特征索引
        self.threshold = 0  # 用于分裂节点的阈值
        self.left = None  # 左子树
        self.right = None  # 右子树


class MyDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # 决策树的最大深度

    def fit(self, X, y):
        self.n_classes_ = len(set(y))  # 类别数量
        self.n_features_ = X.shape[1]  # 特征数量
        self.tree_ = self._grow_tree(X, y)  # 生成决策树

    def predict(self, X):
        # 对输入数据进行预测
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        # 找到最佳分裂点
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        # 使用递归方式生成决策树
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)# 返回数量最多的类别
        node = Node(predicted_class=predicted_class)# 生成节点
    
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # 生成左右子树
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr 
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        # 对单个样本进行预测
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class



if __name__ == '__main__':
    X,y=get_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MyDecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_val)
    print(sum(y_pred==y_val)/len(y_pred))
