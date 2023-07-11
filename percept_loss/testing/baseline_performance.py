import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.testing.benchmark_encodings import random_forest_test
from percept_loss.networks import AUTOENCODER
from percept_loss.losses import LOSS
from percept_loss.utils.savers import train_saver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up parameters
network = 'conv'
print(f'network: {network}')


# DATASETS
_, _, test_dataloader, _ = get_all_loaders(device=device,
                                                                                 batch_size=32)

# used to get the baseline performance
class dummy_encoder:
    def __init__(self):
        self.latent_dim = 32*32*3

    def encoder_forward(self, x):
        return x

acc = random_forest_test(test_dataloader, dummy_encoder(), device)
print(f'Random Forest data Accuracy: {acc*100}')