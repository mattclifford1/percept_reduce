'''
seed everything that affects a single run.

without this, network init and dataloader shuffle order are free-running, so two runs that
differ only in their loss function are not a controlled comparison -- the measured spread of
untrained-network probe accuracy is +/- 0.032 on CIFAR MLP, which is larger than several of
the effects being compared. call set_seed() immediately before building the network.
'''
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
