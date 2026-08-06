from .standard import MSE, MAE
from .perceptual import SSIM_torchmetrics, LPIPS, LPIPS1, SSIM2, MSSIM, NLPD, DISTS

LOSS = {
    'MSE': MSE,
    'MAE': MAE,
    'SSIM_torchmetrics': SSIM_torchmetrics,
    'SSIM': SSIM2,
    'MSSIM': MSSIM,
    'LPIPS': LPIPS,
    'LPIPS1': LPIPS1,   # LPIPS fed [0,1] as if it were [-1,1] -- the pre-fix behaviour
    'NLPD': NLPD,
    'DISTS': DISTS,
}
# N.B do not include '-' in names as inteferes with saving schema