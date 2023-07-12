import torch.nn as nn


def cross_entropy():
    criterion = nn.CrossEntropyLoss()
    return criterion

def MSE():
    criterion = nn.MSELoss()
    return criterion

def MAE():
    criterion = nn.L1Loss()
    return criterion

