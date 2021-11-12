"""
Defines the loss functions used in NeurReg
"""

from typing import Tuple
import torch
import torch.nn.functional as F

__all__ = ["NeurRegLoss"]


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
    use_cuda: bool
        Set to True to use cuda
    """

    def __init__(
        self, λ, β, window_size: Tuple[int, int, int] = (5, 5, 5), use_cuda=False
    ):
        self.λ = λ
        self.β = β
        self.window_size = window_size
        self.use_cuda = use_cuda

    def __call__(
        self, F_0, F_0g, I_0, I_0R, I_1, I_1R, S_0, S_0g, S_1, S_1g
    ) -> torch.Tensor:
        return (
            registration_field_loss(F_0, F_0g)
            + self.λ
            * (
                local_cross_correlation_loss3D(
                    I_0, I_0R, self.window_size, self.use_cuda
                )
                + local_cross_correlation_loss3D(
                    I_1, I_1R, self.window_size, self.use_cuda
                )
            )
            + self.β * (tversky_loss2(S_0, S_0g) + tversky_loss2(S_1, S_1g))
        )


def registration_field_loss(reg_gt: torch.Tensor, reg_pred: torch.Tensor) -> float:
    """
    Computes the registration field loss (L_f)
    """
    # reg_gt, reg_pred = reg_gt.squeeze(), reg_pred.squeeze()
    assert reg_gt.shape == reg_pred.shape

    Ω = torch.prod(torch.tensor(reg_gt.shape))
    return (1 / Ω) * torch.norm(reg_pred - reg_gt)


def local_cross_correlation_loss3D(
    image_gt: torch.Tensor,
    image_pred: torch.Tensor,
    window_size: Tuple[int, int, int],
    use_cuda: bool,
) -> torch.Tensor:
    """
    Computes the local cross correlation loss (L_sim)

    The default window size is the default for the hippocampus dataset
    The loss function only supports 3D inputs.
    """
    assert image_gt.shape == image_pred.shape

    Ω = torch.prod(torch.tensor(image_gt.shape))
    conv_mag = torch.prod(torch.Tensor(window_size)).item()
    kernel = torch.full((1, 1, *window_size), fill_value=(1 / conv_mag))
    if use_cuda:
        kernel = kernel.cuda()
    kernel.requires_grad = False

    gt_mean = F.conv3d(image_gt, kernel, padding=2)
    pred_mean = F.conv3d(image_pred, kernel, padding=2)

    # define constants for shorter formula
    # im_d = image diff
    # p_d  = prediction diff
    im_d, p_d = image_gt - gt_mean, image_pred - pred_mean

    numerator = torch.square(torch.sum((im_d) * (p_d)))
    denominator = torch.sum(torch.square(im_d)) * torch.sum(torch.square(p_d))
    denominator = torch.clamp(denominator, min=1e-5)
    cross_corr = -(1 / Ω) * (numerator / denominator)

    return cross_corr


def tversky_loss2(
    seg_gt: torch.Tensor,
    seg_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the tversky loss for the 2-class case
    """
    seg_sum = torch.sum(seg_gt + seg_pred)
    denominator = torch.clamp(seg_sum, min=1e-5)
    return -1 / 2 * torch.sum(2 * seg_gt * seg_pred) / denominator
