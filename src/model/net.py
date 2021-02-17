import torch
import torch.nn as nn
import torch.nn.init as init

# model definition
class Net(nn.Module):
    # define model elements
    def __init__(self, n_channels):
        super(Net, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Conv2d(n_channels, 32, (3,3))
        init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # first pooling layer
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
        # second hidden layer
        self.hidden2 = nn.Conv2d(32, 32, (3,3))
        init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # second pooling layer
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))
        # fully connected layer
        self.hidden3 = nn.Linear(5*5*32, 100)
        init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()
        # output layer
        self.hidden4 = nn.Linear(100, 10)
        init.xavier_uniform_(self.hidden4.weight)
        self.act4 = nn.Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        X = X.view(-1, 4*4*50)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X

