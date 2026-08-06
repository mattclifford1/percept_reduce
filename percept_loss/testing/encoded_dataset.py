'''
make dataset from the encodings of images
'''
from tqdm import tqdm
import numpy as np
import torch


def make_encodings(data_loader, autoencoder, device):
    '''
    encode a split with the network frozen.

    eval() is load-bearing: in train mode a BatchNorm encoder normalises each batch by its own
    statistics, so an image's encoding would depend on whichever images shared its batch. that
    was harmless while every registered net was norm-free, and silently wrong from the moment
    dcgan/resnet18/vae were added (FINDINGS B9).
    '''
    was_training = autoencoder.training
    autoencoder.eval()
    instances = len(data_loader.dataset)
    X = np.ones((instances, autoencoder.latent_dim))*100
    y = np.ones(instances)*100
    prev_ind = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(device), data[2]
            batch_size = labels.shape[0]
            encodings = autoencoder.encoder_forward(inputs)
            next_ind = prev_ind + batch_size
            encodings = encodings.detach().cpu().numpy()
            encodings = encodings.reshape(batch_size, -1)
            labels = labels.detach().cpu().numpy()

            X[prev_ind:next_ind] = encodings
            y[prev_ind:next_ind] = labels

            prev_ind = next_ind
    autoencoder.train(was_training)
    return X, y
