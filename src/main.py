from model.net import Net
from model.data_loader import(
    instantiate_data,
    instantiate_training_data,
    instantiate_val_data,
    )
import torch
import torch.nn as nn
import torch.optim as optim
from train import training_loop
from evaluate import(
    evaluate,
    evaluate_validation,
    evaluate_training,
    )

def main():

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
     
    model = Net(1).to(device=device)

    data_path = "../Mnist/"

    mnist = instantiate_training_data(data_path)
    mnist_val = instantiate_val_data(data_path)
    
    train_loader = torch.utils.data.DataLoader(mnist, batch_size=64)
    val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=64)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    
    loss_fn = nn.CrossEntropyLoss()

    training_string = "Training"
    val_string = "Val"
    
    print(f"Training on device {device}.")
    
    training_loop(
        n_epochs = 100,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        train_loader = train_loader,
        device = device,
        )
    
    evaluate_training(model, train_loader, training_string)
    evaluate_validation(model, val_loader, val_string)
    
main()

