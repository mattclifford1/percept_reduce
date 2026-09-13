'''
dcgan with GDN in place of BatchNorm and its activations.

same channel plan, same checkerboard-free decoder, same Tanh bottleneck and Sigmoid output as
dcgan_autoencoder.DCGAN_AE, so dcgan vs dcgan_gdn is an ablation of the normalisation alone:
batch statistics (BatchNorm) against a per-sample, perceptually motivated divisive normalisation
(GDN, see gdn.py). in Balle et al.'s networks GDN is itself the nonlinearity, so it replaces the
LeakyReLU/ReLU too; the decoder uses inverse GDN, as learned image codecs do.
'''
import torch.nn as nn

from .dcgan_autoencoder import up_block
from .gdn import GDN


class DCGAN_GDN_AE(nn.Module):
    def __init__(self):
        super(DCGAN_GDN_AE, self).__init__()
        self.latent_dim = 96*2*2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            GDN(12),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            GDN(24),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            GDN(48),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            up_block(96, 48),                                    # [batch, 48, 4, 4]
            GDN(48, inverse=True),
            up_block(48, 24),                                    # [batch, 24, 8, 8]
            GDN(24, inverse=True),
            up_block(24, 12),                                    # [batch, 12, 16, 16]
            GDN(12, inverse=True),
            up_block(12, 3),                                     # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def encoder_forward(self, x):
        return self.encoder(x)

    def decoder_forward(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder_forward(x)
        return self.decoder_forward(z)
