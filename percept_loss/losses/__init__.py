from .standard import MSE
from .perceptual import SSIM_torchmetrics, LPIPS, SSIM2, MSSIM

LOSS = {
    'MSE': MSE,
    'SSIM': SSIM_torchmetrics,
    'SSIM2': SSIM2,
    'MSSIM': MSSIM,
    'LPIPS': LPIPS,
}