import csv
import glob
import os
import random

import pandas as pd
import torch
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

# Giraffes are the best ~  Anne Chow
""" cuda = True if torch.cuda.is_available() else False
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125) """


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

        print(data_orig)
        # converts input data into a tensor
        self.data = torch.Tensor(data_orig)
        
    def keep_columns(self, L):
        """Select Features """           
        feature_data = self.data[:,L]
        feature_data = feature_data.astype(float)
        return feature_data

        
class LSTM_RNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTM_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, song):
        embeds = self.word_embeddings(song)
        x = embeds.view(len(song), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs
        

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)

def train():
    train_data, dev_data, test_data, word_to_ix, label_to_ix = feature_data.load_MR_data()
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 50
    EPOCH = 100
    best_dev_acc = 0.0
    model = LSTM_RNN(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)

    no_up = 0
    for i in range(EPOCH): 
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), 'best_models/mr_best_model_acc_' + str(int(test_acc*10000)) + '.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()


if __name__ == "__main__": 
    # call function to prepare data structure
    pathname = os.getcwd()
    train_data = Music_Data(pathname + '\input_data\\train') 
    val_data = Music_Data(pathname + '\input_data\val')
    test_data = Music_Data(pathname + '\input_data\\test')
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']
    # remove unnecessary columns and convert array to float
    train_feature_data = train_data.keep_columns([0, 1, 3, 6, 9, 10, 18]) # dance, energy, loudness, acousticness, valence, tempo, mood
    #print(train_feature_data[0])
    val_feature_data = val_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    test_feature_data = test_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    #print(feature_data[0:5,:])    
    df_train = pd.DataFrame(train_feature_data, columns = title)
    df_test = pd.DataFrame(test_feature_data, columns = title)
    print(df_train)
    print(df_test)
    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']
