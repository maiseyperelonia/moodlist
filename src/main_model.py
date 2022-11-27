import csv
import glob
import os
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.utils import shuffle
import sklearn.metrics as skm
from sklearn.metrics import classification_report


# check device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

music_train = []
music_test = []
music_val = []
# Load Data

torch.manual_seed(0)
np.random.seed(0)
# Helper Music_Data Class
class Music_Data:
    """Music data set"""

    def __init__(self, path):
        """Music data set"""
        print(path)
        train_path = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']
        data_orig = []
        label_val = []
        num = -1
        # loop over the list of .csv files
        for word in train_path:
            total = path + '\\' + word
            num+=1
            csv_files = glob.glob(os.path.join(total, "*.csv"))
            for file in csv_files:
                # read the csv file
                df = pd.read_csv(file)
                with open(file) as fd:
                    reader = csv.reader(fd, delimiter=',')
                    temp_row = []
                    row_cnt = 0
                    for row in reader:
                        if( row_cnt == 9):
                            break
                        if row[0] == 'danceability':
                            continue
                        col_idx = [0,1,6,9]
                        temp = []
                        for col in col_idx:
                            if(row[col] == "{'status': 401, 'message': 'The access token expired'}"):
                                print(file)
                            
                            temp.append(float(row[col]))
                        temp_row.append(temp)
                        row_cnt += 1
                        # np.append(temp_row, temp)
                    # temp_row = np.array(temp_row)
                    if(row_cnt < 9):
                        while(9 - row_cnt != 0):
                            temp_row.append([0,0,0,0])
                            row_cnt += 1
                            
                    tensor_row = torch.tensor(temp_row)
                    # print("old tensor: ", tensor_row)
                    tensor_row = tensor_row.view(-1,4*9)
                    # print("new tensor: ", tensor_row)
                    # print("tensor as list ", tensor_row.tolist()[0])
                    data_orig.append(tensor_row.tolist()[0])
                    label_val.append(num)

        # data_orig = np.array(data_orig)
        # print(data_orig)
        # label_val = np.array(label_val)
        p = np.random.permutation(len(data_orig))
        data_orig = np.array(data_orig)[p]
        label_val = np.array(label_val)[p]
                    
        self.feat_data = torch.tensor(data_orig.tolist(),
                            dtype=torch.float32).to(device)
        self.label_data = torch.tensor(label_val.tolist(),
                            dtype=torch.float32).to(device)          
        # print(data_orig)
        # print("feature data: " , self.feat_data)
        # self.feat_data = torch.flatten(self.feat_data, start_dim=4, end_dim=9)

        print(self.feat_data.shape)
        # for i in feat_data:
        #     self.feat_data = i.view(-1, 4*9)
        #     print(i)
        #     print("shape flattened:", i.shape)
        # print("label data: ", self.label_data)
        #print("shape of features:" ,self.feat_data.shape)
        # converts input data into a tensor
        # self.data = torch.tensor(data_orig)

    def __len__(self):
        return len(self.feat_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.feat_data
        pol = self.label_data
        
        return self.feat_data[idx], self.label_data[idx]
        
    # def keep_columns(self, L):
    #     """Select Features """           
    #     feature_data = self.feat_data[:,L]
    #     feature_data = feature_data.astype(float)
    #     return feature_data

def readFiles():
    files = glob.glob("./input_data/train/Angry/*")
    files[:10]

# fully connected neural network
class fcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(fcn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(233, hidden_size)
        # self.fc2 = nn.Linear(233, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 23)
        self.fc5 = nn.Linear(23, num_classes)
    
    def forward(self, x):
        x = x.view(-1, input_size) # Flattens input to n x 6 x 1
        # print(x.shape)
        out = F.dropout(F.relu(self.fc1(x)), 0.2)
        # out = self.relu(out)
        # out = F.dropout(F.relu(self.fc2(out)), 0.2)
        # out = F.relu(self.fc3(out))
        out = F.dropout(F.relu(self.fc4(out)), 0.2)
        # out = self.relu(out)
        out = self.fc5(out)
        return out

# 
# def get_accuracy(truth, pred):
#     assert len(truth)==len(pred)
#     right = 0
#     for i in range(len(truth)):
#         if truth[i]==pred[i]:
#             right += 1.0
#     return right/len(truth)

def get_accuracy(model, train=False, val=False, batch_size=64):
    if train:
        print('train')
        data = music_train
    elif val:
        print('val')
        data = music_val
    else:
        print('test')
        data = music_test
        print(music_test)

    correct = 0
    total = 0
    for features, labels in torch.utils.data.DataLoader(data, batch_size=batch_size):
        output = model(features)
        if (data == music_test):
            print(output)
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += features.shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=10 , print_stat = 1, lr= 0.001):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    print("size after dataloader: ", train_loader.dataset.feat_data.shape, "label size: ", train_loader.dataset.label_data.shape)
    criterion = nn.CrossEntropyLoss() # loss function for multi class classification
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []
    # print("The dataLoader:")
    # train_labels = next(iter(train_loader))
    #print(train_labels[0])
    # training
    n = 0 # the number of iterations
    train_err = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        best_loss = 0
        for songs, labels in iter(train_loader):
            # songs = songs.view(-1, 4 * 9)
            # print("song shape: ",songs.shape)
            # print("label shape: ", labels.shape)
            # print(songs)
            out = model(songs)             # forward pass
            loss = criterion(out, labels.long()) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter

            optimizer.zero_grad()         # a clean up step for PyTorch
            
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, train=True)) # compute training accuracy 
            val_acc.append(get_accuracy(model, train=False, val=True))  # compute validation accuracy
            # corr = (out > 0.0).long() != labels.long()
            # total_train_err += int(corr.sum())
            # total_train_loss += loss.item()
            total_epoch += len(labels)
            # save the current training information
            
            n += 1

        # train_err[epoch] = float(total_train_err) / total_epoch
 
        print(("Epoch {}: train_acc: {} val_acc: {}").format(
                   epoch + 1, train_acc[-1], val_acc[-1]))
    if print_stat:
      # plotting
      plt.title("Training Curve")
      plt.plot(iters, losses, label="Train")
      plt.xlabel("Iterations")
      plt.ylabel("Loss")
      plt.show()

      plt.title("Training Curve")
      plt.plot(iters, train_acc, label="Train")
      plt.plot(iters, val_acc, label="Validation")
      plt.xlabel("Iterations")
      plt.ylabel("Training Accuracy")
      plt.legend(loc='best')
      plt.show()

      print("Final Training Accuracy: {}".format(train_acc[-1]))
      print("Final Validation Accuracy: {}".format(val_acc[-1]))

if __name__ == "__main__": 
    # call function to prepare data structure
    pathname = os.getcwd()
    train_data = Music_Data(pathname + '\input_data\\train') 
    val_data = Music_Data(pathname + '\input_data\\val')
    test_data = Music_Data(pathname + '\input_data\\testing_data')
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']

    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']

    # print(train_data)
    music_train = train_data
    music_val = val_data
    music_test = test_data

    # define hyperparameters
    input_size = 36
    hidden_size = 433
    num_classes = 5
    num_epochs = 60
    batch_size = 128
    learning_rate = 0.01
    print("lr: ", learning_rate, "batch_size: ", batch_size, "num_epochs: ", num_epochs)

    model = fcn(input_size, hidden_size, num_classes)
    train(model, train_data, lr= learning_rate, num_epochs=num_epochs)

    print("Testing Accuracy: ", get_accuracy(model, train=False, val=False))

    # sad:
    # 1. carnival town norah jones
    # 2. yanghwa brdg zion t.
    # 3. IF YOU BIGBANG
    # 4. tired adele