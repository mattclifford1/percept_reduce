import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders_CIFAR
from percept_loss.testing.benchmark_encodings import test_all_classifiers
from percept_loss.networks import CIFAR_AUTOENCODERS
from percept_loss.losses import LOSS
from percept_loss.utils.savers import train_saver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up parameters
network = 'conv'
print(f'network: {network}')


# DATASETS
_, _, test_dataloader, _ = get_all_loaders_CIFAR(device=device, batch_size=32)

# used to get the baseline performance
class dummy_encoder:
    def __init__(self):
        self.latent_dim = 32*32*3

    def encoder_forward(self, x):
        return x

acc = test_all_classifiers(test_dataloader, dummy_encoder(), device, verbose=True)
print(f'all classifiers accuracy:\n{acc}')