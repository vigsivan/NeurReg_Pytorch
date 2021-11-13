"""
Defines the loss functions used in NeurReg
"""

from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["NeurRegLoss", "VoxelMorphLoss"]

class VoxelMorphLoss(torch.nn.Module):
    """
    Uses implementations from the VoxelMorph paper instead of my implementations
    (useful for comparison)
    """

    def __init__(
        self, λ, β, window_size: Tuple[int,int,int] = (5,5,5), use_cuda=False
    ):
        self.λ = λ
        self.β = β
        self.ncc = NCC(window_size, use_cuda).loss
        self.dice = Dice().loss
        self.mse = MSE().loss

    def __call__(
        self, F_0, F_0g, I_0, I_0R, I_1, I_1R, S_0, S_0g, S_1, S_1g
    ) -> torch.Tensor:
        return (
            self.mse(F_0, F_0g)
            + self.λ * (self.ncc(I_0, I_0R) + self.ncc(I_1, I_1R))
            + self.β * (self.dice(S_0, S_0g) + self.dice(S_1, S_1g))
        )


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)



class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win: Optional[Tuple[int,int,int]]=None, use_cuda: bool=False):
        self.win = win
        self.use_cuda = use_cuda

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        if self.use_cuda:
            sum_filt = torch.ones([1, 1, *win], device="cuda")
        else:
            sum_filt = torch.ones([1, 1, *win])

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class NeurRegLoss(torch.nn.Module):
    """

    The default window size is the default for the hippocampus dataset as per the paper
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
    if use_cuda:
        kernel = torch.full((1, 1, *window_size), fill_value=(1 / conv_mag), device="cuda")
    else:
        kernel = torch.full((1, 1, *window_size), fill_value=(1 / conv_mag))

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
