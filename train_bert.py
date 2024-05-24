from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from dataset import TextDataSet
from transformers import AdamW
from tqdm import tqdm
from config import Training_Config

def train(model, data_loader_train):
    '''
    完成一个 epoch 的训练
    '''
    sum_true = 0
    sum_loss = 0.0

    #确保模型处于训练模式
    model.train()
    for data in tqdm(data_loader_train):
        # 选取对应批次数据的输入和标签
        batch_x, batch_y = data[0].to(device), data[1].to(device)

        y_hat = model(batch_x).logits
        # print(y_hat)

        loss = loss_function(y_hat, batch_y)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
        sum_true += torch.sum(y_hat == batch_y).float()
        sum_loss += loss.item()

    train_acc = sum_true / len(data_loader_train.dataset)
    train_loss = sum_loss / (len(data_loader_train.dataset) / config.batch_size)

    return train_acc, train_loss

def validation(model,data_loader_valid):
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    sum_true = 0

    model.eval()
    with torch.no_grad():
        for data in data_loader_valid:
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x).logits

            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            sum_true += torch.sum(y_hat == batch_y).float()

        return sum_true / len(data_loader_valid.dataset)




if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = Training_Config()

    # 划分训练集和测试集
    train_dataset = TextDataSet("train.tsv")
    total_samples = len(train_dataset)
    train_size = int(0.8 * total_samples)  # 使用80%的数据作为训练集
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    #构造dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                      lr=config.lr,
                      eps=1e-8
                      )

    for i in range(config.epoch):
        train_loss,train_acc = train(model, train_loader)
        if i % config.num_val == 0:
            val_acc = validation(model, val_loader)
            print(
                f"epoch: {i}, train loss: {train_loss:.4f}, train accuracy: {train_acc * 100:.2f}%, valid accuracy: {val_acc * 100:.2f}%")




