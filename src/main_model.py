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

# check device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

music_train = []
music_test = []
music_val = []
# Load Data
# Helper Music_Data Class
class Music_Data:
    """Music data set"""

    def __init__(self, path):
        """Music data set"""
        print(path)
        train_path = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']
        data_orig = []
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
                    count = 0
                    for row in reader:
                        if row[0] == 'danceability':
                            continue
                        col_idx = [0,1,6,9]
                        temp_row = []
                        for col in col_idx:
                            if(row[col] == "{'status': 401, 'message': 'The access token expired'}"):
                                print(file)

                            temp_row.append(float(row[col]))

                        temp_row.append(num)
                        if count > 0 and row[0] != '':
                            data_orig.append(temp_row)
                        count += 1

        # print(data_orig)
        # converts input data into a tensor
        self.data = list(data_orig)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        # self.data = torch.Tensor(data_orig)
        #print(self.data)
        # self.keep_columns([0, 1, 3, 6, 9, 10, 18])
        
    def keep_columns(self, L):
        """Select Features """           
        feature_data = self.data[:,L]
        feature_data = feature_data.astype(float)
        return feature_data

def readFiles():
    files = glob.glob("./input_data/train/Angry/*")
    files[:10]

# fully connected neural network
class fcn(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(fcn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 4 * 1) # Flattens input to n x 6 x 1
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# def get_accuracy(truth, pred):
#     assert len(truth)==len(pred)
#     right = 0
#     for i in range(len(truth)):
#         if truth[i]==pred[i]:
#             right += 1.0
#     return right/len(truth)

def get_accuracy(model, train=False):
    if train:
        data = music_train
    else:
        data = music_val

    correct = 0
    total = 0
    for features in torch.utils.data.DataLoader(data, batch_size=64):
        labels = features[4]
        output = model(features[:4])
        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += features[:4].shape[0]
    return correct / total

def train(model, data, batch_size=64, num_epochs=1 , print_stat = 1):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss() # loss function for multi class classification
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []
    # print("The dataLoader:")
    # train_labels = next(iter(train_loader))
    #print(train_labels[0])
    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for songs in iter(train_loader):
            for features in songs:
                print("Whole tensor:")
                print(features)
                print("4-part tensor:")
                print(features[:4])
                break
            labels = features[4]
            out = model(torch.tensor(features[:4]))             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            train_acc.append(get_accuracy(model, train=True)) # compute training accuracy 
            val_acc.append(get_accuracy(model, train=False))  # compute validation accuracy
            n += 1

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
    test_data = Music_Data(pathname + '\input_data\\test')
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']

    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']

    # print(train_data)
    music_train = train_data
    music_val = val_data

    # define hyperparameters
    input_size = 9
    hidden_size = 500
    num_classes = 5
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.0001

    model = fcn(input_size, hidden_size, num_classes)
    train(model, train_data)
