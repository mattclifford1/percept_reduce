# adapted from:
# https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py

import torch
import torch.nn as nn
 

class noraml_64(nn.Module):
    def __init__(self):
        super(noraml_64, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.latent_dim = 24*4*4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(6, 10, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(10, 16, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
			nn.Conv2d(16, 24, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
			nn.ConvTranspose2d(16, 10, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(10, 6, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    
    def encoder_forward(self, x):
        return self.encoder(x)
    
    def decoder_forward(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encoder_forward(x)
        return self.decoder_forward(z)
    