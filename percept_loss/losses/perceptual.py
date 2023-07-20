import warnings

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from DISTS_pytorch import DISTS as dists_original

from .NLPD_torch.pyramids import LaplacianPyramid


class sim_to_loss:
    def __init__(self, sim_metric):
        self.sim_metric = sim_metric
    
    def __call__(self, *args, **kwargs):
        return 1 - self.sim_metric(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        self.sim_metric.to(*args, **kwargs)

def LPIPS():
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def SSIM_torchmetrics():
    return sim_to_loss(structural_similarity_index_measure)

def SSIM2():
    return sim_to_loss(SSIM(data_range=1, size_average=True, channel=3))

def MSSIM():
    return sim_to_loss(MS_SSIM(data_range=1, size_average=True, channel=3, win_size=1))

def NLPD(nlpd_k=1):
    with warnings.catch_warnings():    # we don't care about the warnings these give
        warnings.simplefilter("ignore")
        metric = LaplacianPyramid(nlpd_k)
    return metric

def DISTS():
    with warnings.catch_warnings():    # we don't care about the warnings these give
        warnings.simplefilter("ignore")
        metric = dists_original()()
    return metric