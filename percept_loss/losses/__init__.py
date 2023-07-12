from .standard import MSE, MAE
from .perceptual import SSIM_torchmetrics, LPIPS, SSIM2, MSSIM

LOSS = {
    'MSE': MSE,
    'MAE': MAE,
    'SSIM_torchmetrics': SSIM_torchmetrics,
    'SSIM': SSIM2,
    'MSSIM': MSSIM,
    'LPIPS': LPIPS,
}