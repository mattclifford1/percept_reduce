'''
CIFAR-variant ResNet-18 encoder with a mirrored upsampling decoder.

He, Zhang, Ren & Sun (2016), "Deep Residual Learning for Image Recognition", CVPR.
arXiv:1512.03385.

why it is here: the CIFAR ResNet-18 (3x3 stem, stride 1, no max-pool -- the standard small-image
variant) is *the* backbone of the frozen-probe literature. SimCLR (Chen et al. 2020,
arXiv:2002.05709), BYOL (Grill et al. 2020, arXiv:2006.07733) and SimSiam (Chen & He 2021,
arXiv:2011.10566) all report CIFAR-10 numbers from a probe fit on frozen ResNet-18 features.
running our loss axis on the same backbone makes the accuracies in FINDINGS.md comparable to
published self-supervised numbers instead of interpretable only against our own grid.

fixed latent budget: the natural ResNet-18 output is 512x4x4 = 8192, which would hand the probe
21x the input dimension of conv_big_z and win on probe capacity alone (see the capacity note in
README.md). a final stride-2 3x3 conv projects to 96x2x2 = 384, the conv_big_z budget, so an
architecture comparison is not secretly a latent-size comparison.

decoder mirrors the stages with Upsample + 3x3 conv for the same checkerboard reason as
dcgan_autoencoder (Odena, Dumoulin & Olah 2016).

width: `width=64` is the published ResNet-18 (~11M params before the decoder). `width=32` halves
every stage for ~4x less compute and is registered separately as `resnet18_thin` -- the grid is
35 cells and the full-width version is ~100x the parameters of conv_big_z.
'''

import torch.nn as nn


class BasicBlock(nn.Module):
    '''the standard two-3x3 residual block, projection shortcut when shape changes'''
    def __init__(self, in_c, out_c, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def stage(in_c, out_c, blocks, stride):
    layers = [BasicBlock(in_c, out_c, stride=stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_c, out_c, stride=1))
    return nn.Sequential(*layers)


def up_block(in_c, out_c):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResNet18_AE(nn.Module):
    def __init__(self, width=64):
        super(ResNet18_AE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        w = width
        self.latent_dim = 96*2*2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=1, padding=1, bias=False),  # CIFAR stem: 3x3, no max-pool
            nn.BatchNorm2d(w),
            nn.ReLU(),
            stage(w, w, 2, stride=1),                             # [batch, w, 32, 32]
            stage(w, w*2, 2, stride=2),                           # [batch, 2w, 16, 16]
            stage(w*2, w*4, 2, stride=2),                         # [batch, 4w, 8, 8]
            stage(w*4, w*8, 2, stride=2),                         # [batch, 8w, 4, 4]
            nn.Conv2d(w*8, 96, 3, stride=2, padding=1),           # [batch, 96, 2, 2] latent budget
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(96, w*8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w*8),
            nn.ReLU(),
            up_block(w*8, w*4),                                   # [batch, 4w, 4, 4]
            BasicBlock(w*4, w*4),
            up_block(w*4, w*2),                                   # [batch, 2w, 8, 8]
            BasicBlock(w*2, w*2),
            up_block(w*2, w),                                     # [batch, w, 16, 16]
            BasicBlock(w, w),
            up_block(w, w),                                       # [batch, w, 32, 32]
            nn.Conv2d(w, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encoder_forward(self, x):
        return self.encoder(x)

    def decoder_forward(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder_forward(x)
        return self.decoder_forward(z)


def resnet18_AE():
    return ResNet18_AE(width=64)


def resnet18_thin_AE():
    return ResNet18_AE(width=32)
