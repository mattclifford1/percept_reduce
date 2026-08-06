'''
VAE on the DCGAN backbone -- the "deep feature consistent VAE" comparison.

Kingma & Welling (2014), "Auto-Encoding Variational Bayes", ICLR. arXiv:1312.6114.
Hou, Shen, Sun & Qiu (2017), "Deep Feature Consistent Variational Autoencoder", WACV.
arXiv:1610.00291.

why it is here: Hou et al. is the closest published relative of this whole experiment -- a VAE
whose reconstruction term is a VGG feature (perceptual) loss, evaluated on how usable the
latents are. our grid already varies the reconstruction term; adding the KL term gives the
latent a reason to be smooth and lets the writeup say something about *why* perceptual latents
probe better, rather than only that they do.

contract notes:
- `encoder_forward` returns **mu**, not a sample. the probe must see a deterministic encoding,
  otherwise probe accuracy picks up sampling noise. `forward` samples during training and uses
  mu in eval, which is the standard arrangement and keeps val MSE stable across epochs.
- `self.kl` is set by every `forward`; `training/run_and_test.py` adds `beta * net.kl` to the
  loss when the attribute is present. that branch is the only VAE-specific line in the trainer.

**beta is a guess and needs a sweep.** the KL is stored already divided by the pixel count
(3*32*32), so it is on the same per-pixel scale as MSE and lands around 0.01-0.1 in practice --
comparable to the perceptual losses, which return roughly 0.01-1. that makes beta=1 a defensible
starting point and nothing more; the reconstruction/KL balance is exactly the knob that decides
whether the latent is informative or posterior-collapsed. logged as an open item in TODO.md.
'''

import torch
import torch.nn as nn

from percept_loss.networks.dcgan_autoencoder import up_block

PIXELS = 3*32*32


class DCGAN_VAE(nn.Module):
    def __init__(self, beta=1.0):
        super(DCGAN_VAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.latent_dim = 96*2*2
        self.beta = beta
        self.kl = 0.0

        self.features = nn.Sequential(
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
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        # no Tanh on mu -- a bounded mean fights the unit-gaussian prior
        self.to_mu = nn.Linear(96*2*2, self.latent_dim)
        self.to_logvar = nn.Linear(96*2*2, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, 96*2*2)
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

    def encode_dist(self, x):
        h = self.features(x)
        return self.to_mu(h), self.to_logvar(h)

    def encoder_forward(self, x):
        # deterministic encoding for the probe
        mu, _ = self.encode_dist(x)
        return mu

    def decoder_forward(self, z):
        h = self.from_latent(z).reshape(-1, 96, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        # per-pixel KL so it sits on the same scale as the reconstruction losses
        self.kl = -0.5*torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))/PIXELS
        if self.training:
            z = mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
        else:
            z = mu
        return self.decoder_forward(z)
