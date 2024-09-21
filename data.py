import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


def augment_data(x, y, num_augmentations=50):
    _, set_size, _ = x.shape
    x_aug = []
    y_aug = []
    for _ in range(num_augmentations):
        permuted_indices = torch.randperm(set_size)
        x_aug.append(x[:, permuted_indices, :])
        y_aug.append(y)
    x_aug = torch.cat(x_aug)
    y_aug = torch.cat(y_aug)
    return x_aug, y_aug


def get_train_test(train_size, test_size, set_size, data_dim,batch_size, device):
    train_loader = _get_data(train_size, set_size, data_dim,batch_size, device)
    test_loader = _get_data(test_size, set_size, data_dim,batch_size, device)
    return train_loader, test_loader

def _get_data(data_size, set_size, data_dim,batch_size, device):
    x = torch.randn((data_size, set_size, data_dim))
    x[data_size//2:] = x[data_size//2:] * np.sqrt(0.8)
    y = torch.ones(data_size)
    y[data_size // 2:] = -1
    dataset = TensorDataset(x.to(device), y.to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


