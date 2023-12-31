from .standard import MSE, MAE
from .perceptual import SSIM_torchmetrics, LPIPS, SSIM2, MSSIM, NLPD, DISTS

LOSS = {
    'MSE': MSE,
    'MAE': MAE,
    'SSIM_torchmetrics': SSIM_torchmetrics,
    'SSIM': SSIM2,
    'MSSIM': MSSIM,
    'LPIPS': LPIPS,
    'NLPD': NLPD,
    'DISTS': DISTS,
}
# N.B do not include '-' in names as inteferes with saving schema