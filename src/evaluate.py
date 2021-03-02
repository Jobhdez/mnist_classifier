import torch

def evaluate_training(model, train_loader, string):
    """
    calculate the accuracy of the training set.
    """
    evaluate(model, train_loader, string)

def evaluate_validation(model, val_loader, string):
    """
    calculate the accuracy of the validation set.
    """
    evaluate(model, val_loader, string)

def evaluate(model, loader, istraining):
    """
    a general function used to calculate the accuracy of the 
    training and validation set."
   
    Model Loader -> Accuracy
    """
    correct = 0
    total = 0
    for imgs, labels in loader:
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

    print(istraining + " " + "Accuracy: {:.2f}".format(correct / total))

