from torchvision import(
    datasets,
    transforms,
    )

### A DATA_PATH is a string.
### interpretation: a DATA_PATH is the location from which the data
### will downloaded.

### A DATASET is a sub-class of torch.utils.data.Dataset

def instantiate_data(data_path, isTraining):
    """A higher-order procedure that instantiates training or val datasets.

      It transforms each PIL image to a Pytorch Tensor.

      Moreover, it normalizes the data.

      DATA_PATH -> DATASET"""
    return datasets.MNIST(data_path,
                          train=isTraining,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,),
                                                   (0.3081,))]))

def instantiate_training_data(data_path):
    """This function instantiates the training dataset."""
    return instantiate_data(data_path, True)

def instantiate_val_data(data_path):
    """This function instantiates the validation dataset."""
    return instantiate_data(data_path, False)



