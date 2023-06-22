'''
make dataset from the encodings of images
'''
from tqdm import tqdm
import numpy as np


def make_encodings(data_loader, autoencoder, device):
    instances = len(data_loader.dataset)
    X = np.empty((instances, autoencoder.latent_dim))
    y = np.empty(instances)
    prev_ind = 0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data[0].to(device), data[2]
        encodings = autoencoder.encoder_forward(inputs)
        next_ind = prev_ind + labels.shape[0]
        encodings = encodings.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        X[prev_ind:next_ind] = encodings
        y[prev_ind:next_ind] = labels
    return X, y
