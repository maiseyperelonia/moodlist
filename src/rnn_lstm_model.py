import torch
import torchtext
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import Tensor
import torch.nn.functional as F
import torchdata.datapipes as dp
import glob
import os
import pandas as pd
import csv
import numpy as np

import pdb
import math

cuda = True if torch.cuda.is_available() else False
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

# First time run, downloads 823MB file for GloVe
glove = torchtext.vocab.GloVe(name="6B", dim=50) 
# # load data 
# def load_data():
#     train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
#     test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#     batch_size = 100
#     n_iters = 6000
#     num_epochs = n_iters / (len(train_dataset) / batch_size)
#     num_epochs = int(num_epochs)
    
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load Data
# Helper Music_Data Class
# Correct values in Acousticness, Valence, Energy, Danceability, Loudness, Tempo
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
                        row.append(num)
                        if count > 0 and row[0] != '':
                            data_orig.append(row)
                        count += 1
                        for field in row:
                            if field == "{'status': 429, 'message': 'API rate limit exceeded'}":
                                print(file)
        self.data = np.array(data_orig, dtype=object)
        
    def keep_columns(self, L):
        """Select Features """           
        feature_data = self.data[:,L]
        feature_data = feature_data.astype(float)
        return feature_data


# # LSTM cell implementation
# class LSTMCell(nn.Module):

#     def __init__(self, input_size, hidden_size, bias=True):
#         super(LSTMCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
#         self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
#         self.c2c = Tensor(hidden_size * 3)
#         self.reset_parameters()


#     def reset_parameters(self):
#         std = 1.0 / math.sqrt(self.hidden_size)
#         for w in self.parameters():
#             w.data.uniform_(-std, std)
    
#     def forward(self, x, hidden):
#         #pdb.set_trace()
#         hx, cx = hidden
        
#         x = x.view(-1, x.size(1))
        
#         gates = self.x2h(x) + self.h2h(hx)
    
#         gates = gates.squeeze()
        
#         c2c = self.c2c.unsqueeze(0)
#         ci, cf, co = c2c.chunk(3,1)
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
#         ingate = torch.sigmoid(ingate+ ci * cx)
#         forgetgate = torch.sigmoid(forgetgate + cf * cx)
#         cellgate = forgetgate*cx + ingate* torch.tanh(cellgate)
#         outgate = torch.sigmoid(outgate+ co*cellgate)

#         hm = outgate * F.tanh(cellgate)
#         return (hm, cellgate)

# create model class - Based on Lecture Slides
class SpotifyRNN(nn.Module): 

    def __init__(self, input_size, hidden_size, num_class): 
        super(SpotifyRNN, self).__init__() 
        #https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
        #https://nlp.stanford.edu/projects/glove/
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size 
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, num_class)
    
    def forward(self, x): 
        # Look-up the embeddings 
        x = self.emb(x)
        # Set the initial hidden states with zero
        h0 = torch.zeros(1, x.size(0), self.hidden_size) 
        # Initiate cell sate
        c0 = torch.zeros(1, x.size(0), self.hidden_size) 
        # Forward propagate the RNN 
        out, __ = self.rnn(x, (h0, c0))
        # Pass the output of the last step to the classifier
        return self.fc(out[:,-1,:])

if __name__ == "__main__":
    # call function to prepare data structure
    pathname = os.getcwd()
    train_data = Music_Data(pathname + '\input_data\\train') 
    test_data = Music_Data(pathname + '\input_data\\test')
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']
    # remove unnecessary columns and convert array to float
    train_feature_data = train_data.keep_columns([0, 1, 3, 6, 9, 10, 18]) # dance, energy, loudness, acousticness, valence, tempo, mood
    #print(train_feature_data[0])
    #val_feature_data = val_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    test_feature_data = test_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    #print(feature_data[0:5,:])    
    df_train = pd.DataFrame(train_feature_data, columns = title)
    df_test = pd.DataFrame(test_feature_data, columns = title)
    print(df_train)
    print(df_test)
    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']