from model.net import Net
from model.data_loader import(
    instantiate_data,
    instantiate_training_data,
    )
import torch
import torch.nn as nn
import torch.optim as optim
from train import training_loop

def main():

    model = Net(1)

    data_path = "../Mnist/"

    mnist = instantiate_training_data(data_path)
    
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    
    loss_fn = nn.CrossEntropyLoss()

    training_loop(
        n_epochs = 100,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        train_loader = train_loader,
        )

main()

