'''
ViT autoencoder for 32x32 -- the transformer point of comparison.

Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words", ICLR. arXiv:2010.11929.
He, Chen, Xie, Li, Dollar & Girshick (2022), "Masked Autoencoders Are Scalable Vision
Learners", CVPR. arXiv:2111.06377.
width/depth follow ViT-Tiny as used by DeiT (Touvron et al. 2021, arXiv:2012.12877): dim 192,
3 heads. patch 4 at 32px gives 8x8 = 64 tokens, the usual CIFAR ViT setting.

read this before using the numbers -- two honest caveats:

1. **this is not MAE.** MAE's representation quality comes from the *masking* objective
   (reconstruct 75% held-out patches), not from being a transformer. we cannot mask here: the
   whole experiment compares image-level losses (LPIPS, DISTS, SSIM, NLPD), and those are
   defined on a whole image, not on a subset of patches. masking would change the objective
   per-loss and break the comparison. so this is a plain ViT autoencoder with a bottleneck --
   it tests "does a transformer encoder help *this* setup", not "does MAE work".

2. **it is expected to lose, for reasons unrelated to the hypothesis.** ViTs have no
   convolutional inductive bias and are data-hungry; the 1% cell is 240 images. a poor result
   here is evidence about data scale, not about perceptual losses. it is registered because it
   was worth seeing, not because it is a fair fight.

dropout is 0 throughout (small data, and the probe reads a frozen encoder).
'''

import torch
import torch.nn as nn

PATCH = 4
GRID = 8            # 32 / PATCH
DIM = 192
HEADS = 3
ENC_DEPTH = 6
DEC_DEPTH = 4       # MAE uses a deliberately lighter decoder than encoder


def transformer(depth):
    layer = nn.TransformerEncoderLayer(d_model=DIM,
                                       nhead=HEADS,
                                       dim_feedforward=DIM*4,
                                       dropout=0.0,
                                       activation='gelu',
                                       batch_first=True,
                                       norm_first=True)   # pre-LN, stable without warmup
    return nn.TransformerEncoder(layer, num_layers=depth)


class ViT_AE(nn.Module):
    def __init__(self):
        super(ViT_AE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.latent_dim = 96*2*2   # same 384 budget as conv_big_z / dcgan / resnet18

        # encoder: patchify -> +pos -> transformer -> mean-pool -> bottleneck
        self.patch_embed = nn.Conv2d(3, DIM, PATCH, stride=PATCH)     # [batch, DIM, 8, 8]
        self.enc_pos = nn.Parameter(torch.zeros(1, GRID*GRID, DIM))
        self.enc_blocks = transformer(ENC_DEPTH)
        self.enc_norm = nn.LayerNorm(DIM)
        self.to_latent = nn.Linear(DIM, self.latent_dim)
        self.bottleneck_act = nn.Tanh()

        # decoder: latent -> broadcast to tokens -> +pos -> transformer -> per-patch pixels
        self.from_latent = nn.Linear(self.latent_dim, DIM)
        self.dec_pos = nn.Parameter(torch.zeros(1, GRID*GRID, DIM))
        self.dec_blocks = transformer(DEC_DEPTH)
        self.dec_norm = nn.LayerNorm(DIM)
        self.to_pixels = nn.Linear(DIM, PATCH*PATCH*3)
        self.out_act = nn.Sigmoid()

        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    def encoder_forward(self, x):
        z = self.patch_embed(x)                     # [batch, DIM, 8, 8]
        z = z.flatten(2).transpose(1, 2)            # [batch, 64, DIM]
        z = self.enc_blocks(z + self.enc_pos)
        z = self.enc_norm(z).mean(dim=1)            # global average pool over patches
        return self.bottleneck_act(self.to_latent(z))   # [batch, 384]

    def decoder_forward(self, z):
        batch = z.shape[0]
        t = self.from_latent(z).unsqueeze(1)                    # [batch, 1, DIM]
        t = t.expand(batch, GRID*GRID, DIM) + self.dec_pos      # position tells a token which
        t = self.dec_norm(self.dec_blocks(t))                   # patch it is responsible for
        p = self.to_pixels(t)                                   # [batch, 64, PATCH*PATCH*3]
        # [batch, 64, p*p*3] -> [batch, 3, 32, 32]
        p = p.reshape(batch, GRID, GRID, PATCH, PATCH, 3)
        p = p.permute(0, 5, 1, 3, 2, 4)                         # b, c, gh, ph, gw, pw
        p = p.reshape(batch, 3, GRID*PATCH, GRID*PATCH)
        return self.out_act(p)

    def forward(self, x):
        z = self.encoder_forward(x)
        return self.decoder_forward(z)
