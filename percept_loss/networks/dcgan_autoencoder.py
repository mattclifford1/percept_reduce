'''
DCGAN-style conv autoencoder for 32x32.

Radford, Metz & Chintala (2016), "Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks", ICLR. arXiv:1511.06434.

why it is here: this is deliberately the *same* 4x4 stride-2 stack as
cifar_autoencoder.Autoencoder2 (channels 3->12->24->48->96, latent 96*2*2) plus the two things
that paper found necessary for stable training -- BatchNorm after every conv, and LeakyReLU(0.2)
in the encoder / ReLU in the decoder. keeping the channel plan identical makes dcgan vs
conv_big_z a clean ablation of normalisation alone.

the motivating test: three DISTS cells collapse to a constant output inside one epoch and never
recover (FINDINGS B3). a first-epoch collapse to a constant at Adam's default LR is the failure
mode normalisation exists to prevent, so if DISTS survives here the collapse was optimisation,
not the loss -- which is a cheaper answer than the LR sweep in TODO T1.1.

decoder upsamples with Upsample + 3x3 conv rather than ConvTranspose2d. transposed convs leave
checkerboard artefacts (Odena, Dumoulin & Olah 2016, "Deconvolution and Checkerboard Artifacts",
Distill), and a perceptual loss scores those artefacts directly -- with ConvTranspose2d the
artefact would show up in the results as a loss-axis effect.
'''

import torch.nn as nn


def up_block(in_c, out_c):
    # nearest-neighbour resize then 3x3 conv -- the checkerboard-free upsample
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
    )


class DCGAN_AE(nn.Module):
    def __init__(self):
        super(DCGAN_AE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.latent_dim = 96*2*2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.Tanh(),   # no BN on the bottleneck -- keeps it identical to conv_big_z
        )
        self.decoder = nn.Sequential(
            up_block(96, 48),                                    # [batch, 48, 4, 4]
            nn.BatchNorm2d(48),
            nn.ReLU(),
            up_block(48, 24),                                    # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            up_block(24, 12),                                    # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.ReLU(),
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
