import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(SimpleLSTM, self).__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, dropout = .5, num_layers = num_layers)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, X):
        (_, X) = self.lstm(X)
        X = torch.squeeze(self.output(X[0]))
        return X

class BinaryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(BinaryLSTM, self).__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, dropout = .5, num_layers = num_layers)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        (_, X) = self.lstm(X)
        X = torch.squeeze(self.output(X[0]))
        X = self.sigmoid(X)
        return X