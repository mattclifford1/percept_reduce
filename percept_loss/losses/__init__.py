from .standard import MSE
from .perceptual import SSIM, LPIPS

LOSS = {
    'MSE': MSE,
    'SSIM': SSIM,
    'LPIPS': LPIPS,
}