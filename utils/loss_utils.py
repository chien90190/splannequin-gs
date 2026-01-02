#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def knn_local(gaussians, k=5, lambda_w=2000, mask=None):
    """
    Compute local smoothness regularization loss with masking to exclude certain Gaussians.

    Args:
        gaussians (torch.Tensor): Tensor of shape (N, 3), where N is the number of Gaussians.
        k (int): Number of nearest neighbors to consider.
        lambda_w (float): Weight decay factor for distances.
        mask (torch.Tensor): Optional boolean mask of shape (N,), where True means include the Gaussian.

    Returns:
        torch.Tensor: Smoothness regularization loss.
    """
    if mask is None:
        mask = torch.ones(gaussians.shape[0], dtype=torch.bool, device=gaussians.device)
    active_gaussians = gaussians[mask]  # Shape: (M, 3), where M is the number of valid Gaussians

    # Compute pairwise distances for active Gaussians only
    pairwise_distances = torch.cdist(active_gaussians, active_gaussians, p=2)

    # Get the k nearest neighbors (excluding self)
    _, indices = torch.topk(pairwise_distances, k=k+1, largest=False)
    indices = indices[:, 1:]  # Exclude self from neighbors

    # Compute weights based on distances
    distances = torch.gather(pairwise_distances, 1, indices)
    weights = torch.exp(-lambda_w * distances**2)

    # Compute smoothness loss
    diff = active_gaussians.unsqueeze(1) - active_gaussians[indices]
    return torch.mean(weights * torch.sum(diff**2, dim=-1))


def knn_combined_loss(gaussians, k=5, lambda_local=1.0, lambda_global=0.1, lambda_w=2000):
    """
    --------------------- Memeory intenstive ---------------------
    Combined local and global KNN-based smoothness loss.
    Args:
        gaussians: Tensor of shape (N, 3), positions of Gaussians.
        k: Number of neighbors for local smoothness.
        lambda_local: Weight for local smoothness loss.
        lambda_global: Weight for global smoothness loss.
        lambda_w: Scaling factor for weight computation.
    Returns:
        Combined loss (scalar).
    """
    N = gaussians.shape[0]
    if N <= 1:  # Handle edge case for small number of Gaussians
        return torch.tensor(0.0, device=gaussians.device)

    # Compute pairwise distances
    pairwise_distances = torch.cdist(gaussians, gaussians, p=2)
    
    # Local KNN
    _, indices_local = torch.topk(pairwise_distances, k=k+1, largest=False)
    indices_local = indices_local[:, 1:]  # Exclude self
    distances_local = torch.gather(pairwise_distances, 1, indices_local)
    weights_local = torch.exp(-lambda_w * distances_local**2)
    diff_local = gaussians.unsqueeze(1) - gaussians[indices_local]
    local_loss = torch.mean(weights_local * torch.sum(diff_local**2, dim=-1))
    
    # Global smoothness
    weights_global = torch.exp(-lambda_w * pairwise_distances**2)
    diff_global = gaussians.unsqueeze(1) - gaussians.unsqueeze(0)
    global_loss = torch.mean(weights_global * torch.sum(diff_global**2, dim=-1))
    
    # Combine losses with weights
    combined_loss = lambda_local * local_loss + lambda_global * global_loss
    return combined_loss


def knn_top_percent(points, k=5, top_percent=10, lambda_penalty=100.0, method='cutoff', lambda_w=2000):
    """
    Optimized adaptive KNN loss with top percentage penalization.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3), representing Gaussian coordinates.
        k (int): Number of nearest neighbors to consider.
        top_percent (float): Percentage (0-100) of distances to penalize.
        lambda_penalty (float): Penalty weight for large distances.

    Returns:
        torch.Tensor: Adaptive KNN loss.
    """
    # Compute pairwise distances efficiently using torch.cdist
    pairwise_distances = torch.cdist(points, points, p=2)
    
    # Get k-nearest neighbors
    knn_distances, indices = torch.topk(pairwise_distances, k=k+1, largest=False)

    if method == 'cutoff':
        # Determine the cutoff for the top percentage of distances
        cutoff_distance = torch.quantile(knn_distances, 1 - top_percent / 100.0) + 1e-8
        penalties = torch.clamp(knn_distances - cutoff_distance, min=0).pow_(2)
        return (penalties * lambda_penalty).mean()
    
    elif method == 'common':
        # Compute weights based on distances
        indices = indices[:, 1:]  # Exclude self from neighbors
        distances = torch.gather(pairwise_distances, 1, indices)
        weights = torch.exp(-lambda_w * distances**2)
        diff = points.unsqueeze(1) - points[indices]
        return torch.mean(weights * torch.sum(diff**2, dim=-1))
    else:
        return 0.0
    