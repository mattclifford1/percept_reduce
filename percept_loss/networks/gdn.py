'''
generalised divisive normalisation (GDN) and its inverse.

Balle, Laparra & Simoncelli (2016), "Density Modeling of Images Using a Generalized
Normalization Transformation", ICLR. arXiv:1511.06281.

    y[i] = x[i] / sqrt(beta[i] + sum_j gamma[j, i] * x[j]^2)

why it is here: GDN is the normalisation of learned image compression and of NLPD (one of the
losses in this grid) -- a divisive normalisation motivated by models of early vision rather than
by optimisation. unlike BatchNorm it is per-sample: there are no batch or running statistics, so
a GDN encoder behaves identically in train and eval mode, and a net trained on uniform noise
carries no noise statistics into the probe (compare testing/adabn_reprobe.py).

adapted from CompressAI (InterDigital; compressai/layers/gdn.py, compressai/ops/parametrizers.py,
compressai/ops/bound_ops.py) by way of ~/projects/H-Test-IQM/h_test_IQM/models/utils.py, so that
compressai is not a dependency of this pinned environment.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    '''max(x, bound) whose gradient passes through when it would move x towards the bound'''

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer('bound', torch.Tensor([float(bound)]))

    def forward(self, x):
        return LowerBoundFunction.apply(x, self.bound)


class NonNegativeParametrizer(nn.Module):
    '''keeps beta and gamma non-negative during training, which GDN needs to stay stable'''
    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset**2
        self.register_buffer('pedestal', torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        return out**2 - self.pedestal


class GDN(nn.Module):
    '''GDN, or its inverse (IGDN) with inverse=True -- the decoder side of a learned codec'''

    def __init__(self, in_channels: int, inverse: bool = False, beta_min: float = 1e-6,
                 gamma_init: float = 0.1):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam = NonNegativeParametrizer(minimum=float(beta_min))
        self.beta = nn.Parameter(self.beta_reparam.init(torch.ones(in_channels)))
        self.gamma_reparam = NonNegativeParametrizer()
        self.gamma = nn.Parameter(self.gamma_reparam.init(float(gamma_init)*torch.eye(in_channels)))

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma).reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
        return x*norm
