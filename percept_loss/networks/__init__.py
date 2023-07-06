from .simple_autoencoder import auto_encoder
from .linear_autoencoder import linear_AE
from .cifar_autoencoder import Autoencoder


AUTOENCODER = {
    # 'simple_conv': auto_encoder,  # not working
    # 'linear': linear_AE,    # not working
    'conv': Autoencoder,
}

CLASSIFIER = {
    
}