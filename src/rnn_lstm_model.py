import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import Tensor
import torch.nn.functional as F

import pdb
import math

cuda = True if torch.cuda.is_available() else False
    
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor    

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)

    

# load data 
def load_data():
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    batch_size = 100
    n_iters = 6000
    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# LSTM cell implementation
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        #pdb.set_trace()
        hx, cx = hidden
        
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
    
        gates = gates.squeeze()
        
        c2c = self.c2c.unsqueeze(0)
        ci, cf, co = c2c.chunk(3,1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate+ ci * cx)
        forgetgate = torch.sigmoid(forgetgate + cf * cx)
        cellgate = forgetgate*cx + ingate* torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate+ co*cellgate)

        hm = outgate * F.tanh(cellgate)
        return (hm, cellgate)

# create model class
class SpotifyRNN(nn.Module): 
    def __init__(self, input_size, hidden_size, num_class): 
        super(SpotifyRNN, self).__init__() 
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size 
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, num_class)
    def forward(self, x): 
        # Look-up the embeddings 
        x = self.emb(x)
        # Set the initial hidden states 
        h0 = torch.zeros(1, x.size(0), self.hidden_size) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size) 
        # Forward propagate the RNN 
        out, __ = self.rnn(x, (h0, c0))
        # Pass the output of the last step to the classifier
        return self.fc(out[:,-1,:])
