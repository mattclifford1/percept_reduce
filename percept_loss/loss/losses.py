import torch.nn as nn


def cross_entropy():
    criterion = nn.CrossEntropyLoss()
    return criterion

def MSE():
    criterion = nn.MSELoss()
    return criterion