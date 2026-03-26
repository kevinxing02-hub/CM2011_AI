# data_utils.py
import torch
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name='ptbxl', batch_size=32):
    # Load the .pt files we saved earlier
    test_data = torch.load(f'used_datad/data_basic_ptb/{dataset_name}_test_250.pt')
    
    # Reconstruct variables
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    classes = train_data['classes']
    
    # Create Datasets and Loaders
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, test_loader, len(classes), classes