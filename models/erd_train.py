'''
Dataset:
/home/royliu/Documents/dataset/traffic:
    ├─ empty_road
    │  ├─ 0_empty_0.jpg
    │  ├─ 0_empty_1.jpg
    │  └─ ...
    └─ none_empty_road
       ├─ 0_none_empty_0.jpg
       ├─ 0_none_empty_1.jpg
       └─ ...

'''
## Train ERD

# from tqdm.notebook import tqdm
from tqdm import tqdm, trange
import time
import os
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler  # DO NOT import torch.utils.data.Dataset as Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # SGD, RMSprop, adam, etc
import cv2
import numpy as np
import glob
import pandas as pd
from torch.utils.data.sampler import *
import torchvision.transforms as transforms  # process image
from erd_cnn import NET

random_seed = 42
root = '/home/royliu/Documents/dataset/traffic'
# root = 'C:/dataset/traffic'  # windows @home
data_table = pd.read_csv(os.path.join(root, 'data_table.csv'), index_col=0)
data_table = data_table.sample(frac=1, random_state=random_seed)  # shuffle the rows
data_table = data_table.reset_index(drop=True)  # rest index in order
img_path_list, label_list = data_table.iloc[:, 0], data_table.iloc[:, 1]  # column 0 is image path, column 1 is label


class Load(Dataset):
    def __init__(self, img_path_list, label_list, transform=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform
        pass

    def __getitem__(self, index):
        img = cv2.imread(self.img_path_list[index])  # H, W, C (BGR)
        label = float(self.label_list[index])
        if transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x, y) in tqdm(iterator, desc="Training", leave=False):
        x = x.to(device)
        #         y = y.to(device)
        y = y.type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            # y = y.to(device)
            y = y.type(torch.LongTensor).to(device)
            y_pred, _ = model(x)

            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


##  Training starts here

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    model = NET().to(device)

    ## 1. training data processing
    process_size = (360, 640)
    train_ratio = 0.6
    # valid_ratio = 0.6+0.2
    test_ratio = 0.2
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(process_size)])  # value change to 0 to 1
    dataset = Load(img_path_list, label_list,
                   transform=transform)  # dataset[156][0]: H, W, C (BGR), dataset[156][1]: label, 0 or 1

    indices = list(range(len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    indices_train = indices[: int(train_ratio * len(dataset))]
    indices_valid = indices[int(train_ratio * len(dataset)):-int(test_ratio * len(dataset))]
    indices_test = indices[-int(test_ratio * len(dataset)):]
    sampler_train = SequentialSampler(indices_train)
    sampler_valid = SequentialSampler(indices_valid)
    sampler_test = SequentialSampler(indices_test)

    dataset_train = DataLoader(dataset=dataset, batch_size=24, num_workers=6, shuffle=False, sampler=sampler_train)
    dataset_valid = DataLoader(dataset=dataset, batch_size=24, num_workers=6, shuffle=False, sampler=sampler_valid)
    dataset_test = DataLoader(dataset=dataset, batch_size=16, num_workers=6, shuffle=False, sampler=sampler_test)

    ## 2. training setting
    EPOCHS = 20  # suggest 20 epoch, to achive 98%
    FOUND_LR = 2e-3
    criterion = nn.CrossEntropyLoss().to(device)
    params = [
        {'params': model.features.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.classifier.parameters()}
    ]
    optimizer = optim.Adam(params, lr=FOUND_LR)
    best_valid_loss = float('inf')

    # for epoch in trange(EPOCHS, desc= 'Epochs',ncols=80):  ## in pycharm
    for epoch in tqdm(range(EPOCHS)):  ## in jupyter notebook
        start_time = time.monotonic()

        train_loss, train_acc = train(model, dataset_train, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, dataset_valid, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './erd_20221214.pt')
        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        # torch.save(model.state_dict(), os.path.join(root,'erd_last.pt'))

