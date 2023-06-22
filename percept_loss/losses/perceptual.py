from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure


def LPIPS():
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def SSIM():
    return structural_similarity_index_measure