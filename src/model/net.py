import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    """A representation of a convolutional neural network comprised of VGG blocks."""
    def __init__(self, n_channels):
        super(Net, self).__init__()
        # VGG block 1
        self.conv1 = nn.Conv2d(n_channels, 64, (3,3))
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.dropout = nn.Dropout(0.2)
        # VGG block 2
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.dropout2 = nn.Dropout(0.2)
        # VGG block 3
        self.conv3 = nn.Conv2d(64, 128, (3,3))
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.dropout3 = nn.Dropout(0.2)
        # Fully connected layer
        self.f1 = nn.Linear(128 * 1 * 1, 1000)
        self.dropout4 = nn.Dropout(0.5)
        self.act4 = nn.ReLU()
        # Output layer
        self.f2 = nn.Linear(1000, 10)
        self.act5 = nn.Softmax(dim=1)

    def forward(self, X):
        """This function forward propagates the input."""
        # VGG block 1
        X = self.conv1(X)
        X = self.act1(X)
        X = self.pool1(X)
        X = self.dropout(X)
        # VGG block 2
        X = self.conv2(X)
        X = self.act2(X)
        X = self.pool2(X)
        X = self.dropout2(X)
        # VGG block 3
        X = self.conv3(X)
        X = self.act3(X)
        X = self.pool3(X)
        X = self.dropout3(X)
        # Flatten
        X = X.view(-1, 128)
        # Fully connected layer
        X = self.f1(X)
        X = self.act4(X)
        X = self.dropout4(X)
        # Output layer
        X = self.f2(X)
        X = self.act5(X)

        return X
        
        
