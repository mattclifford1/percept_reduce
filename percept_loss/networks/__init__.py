from .simple_autoencoder import auto_encoder
from .linear_autoencoder import linear_AE
from .cifar_autoencoder import Autoencoder, Autoencoder2, Autoencoder_small


AUTOENCODER = {
    # 'simple_conv': auto_encoder,  # not working
    # 'linear': linear_AE,    # not working
    'conv_small_z': Autoencoder,
    'conv_big_z': Autoencoder2,
    'conv_bigger_z': Autoencoder_small,

}

CLASSIFIER = {
    
}