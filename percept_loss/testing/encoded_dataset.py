'''
make dataset from the encodings of images
'''
from tqdm import tqdm
import numpy as np


def make_encodings(data_loader, autoencoder, device):
    instances = len(data_loader.dataset)
    X = np.ones((instances, autoencoder.latent_dim))*100
    y = np.ones(instances)*100
    prev_ind = 0
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
    return X, y
