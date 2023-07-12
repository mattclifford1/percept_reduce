from .simple_autoencoder import auto_encoder
from .linear_autoencoder import linear_AE
from .cifar_autoencoder import Autoencoder_mini, Autoencoder2, Autoencoder_small, Autoencoder_big


AUTOENCODER = {
    # 'simple_conv': auto_encoder,  # not working
    # 'linear': linear_AE,    # not working
    'conv_small_z': Autoencoder_mini,
    'conv_big_z': Autoencoder2,
    'conv_bigger_z': Autoencoder_small,
    'conv_biggest_z': Autoencoder_big,

}

CLASSIFIER = {
    
}