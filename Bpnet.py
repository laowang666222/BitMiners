import numpy as np
from scipy.spatial import distance
from collections import Counter

from sklearn.model_selection import train_test_split
from utils import transform_data, text_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


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

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == '__main__':
    X,y=get_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # 将数据转换为torch的Tensor
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val, dtype=torch.float32)
    y_val_torch = torch.tensor(y_val, dtype=torch.float32)

    # 创建数据加载器
    train_data = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_data, batch_size=32)

    # 实例化模型
    model = Net()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters())

    import matplotlib.pyplot as plt

    # 初始化列表以存储每个epoch的损失
    train_losses = []
    val_losses = []

    for epoch in range(5):
        # 训练阶段
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())  # 存储训练损失

        # 验证阶段
        val_outputs = model(X_val_torch)
        val_loss = criterion(val_outputs.squeeze(), y_val_torch)
        val_losses.append(val_loss.item())  # 存储验证损失

        val_preds = (val_outputs > 0.5).float().squeeze()
        val_accuracy = (val_preds == y_val_torch).float().mean().item()

        print(f'Epoch {epoch+1}, validation loss: {val_loss.item()}, validation accuracy: {val_accuracy}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


