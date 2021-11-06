"""
Defines the loss functions used in NeurReg
"""

from typing import Tuple
import torch
import torch.nn.functional as F

__all__ = [ "NeurRegLoss" ]

class NeurRegLoss(torch.nn.Module):
    """

    The default window size is the default for the hippocampus dataset
    The loss function only supports 3D inputs.

    Parameters:
    ----------
    λ: float
        Weighting for the similarity (cross correlation) loss
    β: float
        Weighting for the segmentation loss
    window_size: Tuple[int,int,int]
        Window size for computing the cross-correlation
    """
    def __init__(self, λ, β, window_size: Tuple[int,int,int] = (5,5,5)):
        self.λ = λ
        self.β = β
        self.window_size = window_size

    def __call__(
        self,
        registrations: Tuple[torch.Tensor, torch.Tensor],
        images: Tuple[torch.Tensor, torch.Tensor],
        segmentations: Tuple[torch.Tensor, torch.Tensor]
        )->float:
        return (registration_field_loss(*registrations)
                + self.λ *  local_cross_correlation_loss3D(*images, self.window_size)
                + self.β * tversky_loss2(*segmentations))



def registration_field_loss(reg_gt: torch.Tensor, reg_pred: torch.Tensor)->float:
    """
    Computes the registration field loss (L_f)
    """
    # reg_gt, reg_pred = reg_gt.squeeze(), reg_pred.squeeze()
    assert reg_gt.shape == reg_pred.shape

    Ω = torch.prod(torch.tensor(reg_gt.shape)).item()
    return (1 / Ω) * torch.norm(reg_pred - reg_gt)


def local_cross_correlation_loss3D(
    image_gt: torch.Tensor,
    image_pred: torch.Tensor,
    window_size: Tuple[int, int, int]
)-> torch.Tensor:
    """
    Computes the local cross correlation loss (L_sim)

    The default window size is the default for the hippocampus dataset
    The loss function only supports 3D inputs.
    """
    assert image_gt.shape == image_pred.shape

    Ω = torch.prod(torch.tensor(image_gt.shape)).item()
    conv_mag = torch.prod(torch.Tensor(window_size)).item()
    kernel = torch.full((1,1,*window_size), fill_value=(1/conv_mag))

    gt_mean = F.conv3d(image_gt, kernel)
    pred_mean = F.conv3d(image_pred, kernel)

    # define constants for shorter formula
    # im_d = image diff
    # p_d  = prediction diff
    im_d, p_d = image_gt-gt_mean, image_pred-pred_mean

    numerator = torch.square(torch.sum((im_d)*(p_d))).item()
    denominator = torch.sum(torch.square(im_d))*torch.sum(torch.square(p_d)).item()
    cross_corr = -(1/Ω) * (numerator/denominator)

    return cross_corr

def tversky_loss2(
    seg_gt: torch.Tensor,
    seg_pred: torch.Tensor,
    )->torch.Tensor:
    """
    Computes the tversky loss for the 2-class case
    """
    return -1/2 * torch.sum((2*seg_gt*seg_pred)/(seg_gt+seg_pred))
