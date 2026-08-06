from .simple_autoencoder import auto_encoder
from .linear_autoencoder import linear_AE
from .cifar_autoencoder import Autoencoder_mini, Autoencoder2, Autoencoder_small, Autoencoder_big
from .image_net_64_autoencoder import noraml_64, big_64
from .dcgan_autoencoder import DCGAN_AE
from .resnet_autoencoder import resnet18_AE, resnet18_thin_AE
from .vit_autoencoder import ViT_AE
from .vae import DCGAN_VAE


CIFAR_AUTOENCODERS = {
    # 'simple_conv': auto_encoder,  # not working
    # 'linear': linear_AE,    # not working
    'conv_small_z': Autoencoder_mini,
    'conv_big_z': Autoencoder2,
    'conv_bigger_z': Autoencoder_small,
    'conv_biggest_z': Autoencoder_big,
    # literature backbones -- all share conv_big_z's 384 latent budget so that comparing them
    # is not secretly comparing probe input size. see README.md for citations and reasoning.
    'dcgan': DCGAN_AE,              # conv_big_z + BatchNorm/LeakyReLU (Radford et al. 2016)
    'resnet18': resnet18_AE,        # CIFAR ResNet-18, the frozen-probe standard (He et al. 2016)
    'resnet18_thin': resnet18_thin_AE,   # half width, ~4x cheaper
    'vit': ViT_AE,                  # ViT-Tiny AE, expected to lose on data -- see README
    'vae': DCGAN_VAE,               # KL term on the dcgan backbone (Kingma & Welling 2014)
}
# N.B do not include '-' in names as inteferes with saving schema

IMAGENET64_AUTOENCODERS = {
    'standard': noraml_64,
    'bigger_z': big_64,
}

CLASSIFIER = {
    
}