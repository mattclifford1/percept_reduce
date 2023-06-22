import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure


def cross_entropy():
    criterion = nn.CrossEntropyLoss()
    return criterion

def MSE():
    criterion = nn.MSELoss()
    return criterion

def LPIPS():
    return LearnedPerceptualImagePatchSimilarity(net_type='vgg')

def SSIM():
    return structural_similarity_index_measure
