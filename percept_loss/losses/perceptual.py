import warnings
from functools import partial

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from DISTS_pytorch import DISTS as dists_original

from .NLPD_torch.pyramids import LaplacianPyramid

# all data is normalised to [0, 1] (see datasets/torch_loaders.NORMALISE) and every decoder
# ends in a Sigmoid, so every metric here must be correct on that range.

# MS-SSIM scale weights. only 2 scales are usable at 32x32 (3 scales gives NaN), so we use 2
# everywhere to keep CIFAR and ImageNet64 comparable. 64x64 could support 3.
MSSIM_BETAS = (0.5, 0.5)


class sim_to_loss:
    def __init__(self, sim_metric):
        self.sim_metric = sim_metric

    def __call__(self, *args, **kwargs):
        return 1 - self.sim_metric(*args, **kwargs)

    def to(self, *args, **kwargs):
        # functional metrics have nothing to move to a device
        if hasattr(self.sim_metric, 'to'):
            self.sim_metric.to(*args, **kwargs)

def LPIPS():
    # normalize=True tells LPIPS that inputs are [0, 1]; it rescales to the [-1, 1] that the
    # VGG backbone was calibrated on. see LPIPS1 for the uncorrected variant.
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)

def LPIPS1():
    # legacy/uncorrected LPIPS: normalize=False means LPIPS assumes inputs are already
    # [-1, 1], but it is handed [0, 1] data, so only half the intended dynamic range is used.
    # kept as a named loss so the effect of the fix can be measured directly against LPIPS.
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False)

def SSIM_torchmetrics():
    return sim_to_loss(structural_similarity_index_measure)

def SSIM2():
    return sim_to_loss(SSIM(data_range=1, size_average=True, channel=3))

def MSSIM():
    # pytorch_msssim's MS_SSIM hard-asserts min(H,W) > (win_size-1)*2**4, i.e. 161px at the
    # default win_size=11, so it can only run at 32px with win_size=1 -- which collapses the
    # gaussian window to a single pixel and destroys the structural term (~300x weaker
    # gradient). torchmetrics has no such assert and works correctly at 32px.
    return sim_to_loss(partial(multiscale_structural_similarity_index_measure,
                               data_range=1.0,
                               betas=MSSIM_BETAS))

def NLPD(nlpd_k=1):
    with warnings.catch_warnings():    # we don't care about the warnings these give
        warnings.simplefilter("ignore")
        metric = LaplacianPyramid(nlpd_k)
    return metric

class DISTS():
    def __init__(self):
        with warnings.catch_warnings():    # we don't care about the warnings these give
            warnings.simplefilter("ignore")
            self.metric = dists_original()

    def to(self, *args, **kwargs):
        self.metric.to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.metric(*args, **kwargs, require_grad=True, batch_average=True)
