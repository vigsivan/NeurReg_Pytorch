"""
Runs inference with the trained NeurReg model
"""

import sys
import SimpleITK as sitk
from dataset import ImageDataset
from components import *
from typing import Dict
from torch.nn import Conv3d, Sequential, Softmax, Module, Parameter
from torch.distributions import Normal
# import params
import torch

from losses import NeurRegLoss, VoxelMorphLoss
from torch.utils.data import DataLoader


def get_dataloader(params) -> DataLoader:
    dataset = ImageDataset(
        params.path_to_images,
        params.path_to_segs,
        params.matching_fn,
        target_shape=params.target_shape,
    )

    dl = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    use_cuda = "cuda" in params.device
    if use_cuda:
        dl.pin_memory = True

    return dl


def get_untrained_models(params) -> Dict[str, Module]:

    N = Unet3D(inshape=params.target_shape)
    use_cuda = "cuda" in params.device
    if params.loss_func == "nr":
        loss_func = NeurRegLoss(
            params.cross_corr_loss_weight, params.seg_loss_weight, use_cuda=use_cuda
        )
    elif params.loss_func == "vm":
        loss_func = VoxelMorphLoss(
            params.cross_corr_loss_weight, params.seg_loss_weight, use_cuda=use_cuda
        )
    else:
        raise Exception(f"Loss function {params.loss_func} not defined")

    stn = SpatialTransformer(params.target_shape)

    conv_w_softmax = Sequential(Conv3d(17, 1, 3, padding=1), Softmax(3))

    # Copy strategy from voxelmorph
    to_flow_field = Conv3d(16, 3, 3, padding=1, bias=True)
    to_flow_field.weight = Parameter(Normal(0, 1e-5).sample(to_flow_field.weight.shape))
    to_flow_field.bias = Parameter(torch.zeros(to_flow_field.bias.shape))

    if "cuda" in params.device and params.num_gpus > 1:
        N = torch.nn.DataParallel(N)
        N.state_dict = N.module.state_dict

    conv_w_softmax.eval()
    N.eval()

    return {
        "N": N,
        "to_flow_field": to_flow_field,
        "conv_w_softmax": conv_w_softmax,
        "stn": stn,
        "loss_func": loss_func,
    }


def infer(params, data, stn, to_flow_field, conv_w_softmax, N):
    for category in ("moving", "another", "transform"):
        for tensor in ("image", "seg"):
            data[category][tensor] = data[category][tensor].to(params.device)
        if category == "transform":
            data[category]["field"] = data[category]["field"].to(params.device)
            data[category]["concat"] = data[category]["concat"].to(params.device)
        if category == "another":
            data[category]["concat"] = data[category]["concat"].to(params.device)

    last_layer = N(data["transform"]["concat"])
    F_0 = to_flow_field(last_layer)
    F_1 = to_flow_field(N(data["another"]["concat"]))

    #########################
    # Compute loss. Because we have a lot of variables, we'll use
    # the notation from the paper
    #########################
    F_0g = data["transform"]["field"]
    I_0 = data["transform"]["image"]
    I_1 = data["another"]["image"]
    I_0R = stn(data["moving"]["image"], F_0)
    I_1R = stn(data["moving"]["image"], F_1)
    S_0g = data["transform"]["seg"]
    S_0 = stn(data["moving"]["seg"], F_0)
    S_1 = stn(data["moving"]["seg"], F_1)
    S_1g = data["another"]["seg"]
    boosted = torch.cat((last_layer, S_0), 1)
    S_0feat = conv_w_softmax(boosted)

    return (F_0, F_1, F_0g), (
        data["moving"]["image"],
        data["another"]["image"],
        data["transform"]["image"],
        I_0,
        I_1,
        I_0R,
        I_1R,
        S_0g,
        S_0,
        S_1,
        S_1g,
        S_0feat,
    )


def save_images(images, fields):
    field_names = ["learned_field_synthetic", "learned_field_real", "synthetic_field"]

    for f, fname in zip(fields, field_names):
        image_sitk = sitk.GetImageFromArray(f.detach().numpy())
        sitk.WriteImage(image_sitk, f"{fname}.nii.gz")

    imnames = [
        "moving_image",
        "another_image",
        "transformed_image",
        "",
        "",
        "moving_image_transformed_with_learned_field_synthetic",
        "moving_image_transformed_with_learned_field_real",
        "transform_seg",
        "",
        "moving_seg_transformed_real",
        "another_seg",
        "moving_seg_transforemed_synthetic",
    ]

    for i, iname in zip(images, imnames):
        image_sitk = sitk.GetImageFromArray(i.detach().numpy())
        if iname == "":
            continue
        sitk.WriteImage(image_sitk, f"{iname}.nii.gz")


# TODO: modify script to use argparse. Params file is dead!!!!
# if __name__ == "__main__":
#     if len(sys.argv) != 2 or sys.argv[1].lower() not in ("slurm", "cpu"):
#         print(
#             f"Usage: {sys.argv[0]} <config_name>\nwhere <config_name> is one of (slurm, cpu)"
#         )
#         exit(0)
#
#     config = sys.argv[1]
#     if config == "slurm":
#         print("Using SLURM CONFIG")
#         # main(params.SLURM_CONFIG)
#         pms = params.SLURM_CONFIG
#     else:
#         print("Using CPU CONFIG")
#         pms = params.CPU_CONFIG
#         # main(params.CPU_CONFIG)
#
#     models = get_untrained_models(pms)
#     if pms == params.CPU_CONFIG:
#         checkpoint = torch.load(
#             ("/Users/vigsivan/logging/checkpoint.pt"), map_location=torch.device("cpu")
#         )
#     else:
#         checkpoint = torch.load(("/Users/vigsivan/logging/checkpoint.pt"))
#
#     models["N"].load_state_dict(checkpoint["N"])
#     models["to_flow_field"].load_state_dict(checkpoint["to_flow_field"])
#     models["conv_w_softmax"].load_state_dict(checkpoint["conv_w_softmax"])
#
#     val_iter = iter(get_dataloader(pms))
#     data = next(val_iter)
#
#     fields, images = infer(
#         pms,
#         data,
#         models["stn"],
#         models["to_flow_field"],
#         models["conv_w_softmax"],
#         models["N"],
#     )
#
#     save_images(images, fields)

# moving = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("./niis/moving_image.nii.gz"))[0,...]).unsqueeze(0)
# works = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("./niis/fields/synthetic_field.nii.gz")))[0,...].unsqueeze(0)
# does_not_work = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage("./niis/fields/learned_field_synthetic.nii.gz")))[0,...].unsqueeze(0)
# rfloss = registration_field_loss
#
# size = (128,128,128)
# vectors = [torch.arange(0, s) for s in size]
# grids = torch.meshgrid(vectors, indexing="ij")
# grid = torch.stack(grids, dim=0)
# grid = grid.unsqueeze(0)
# grid = grid.float()
#
# def get_new_locs(src, flow):
#     new_locs = grid + flow
#     shape = flow.shape[2:]
#     # need to normalize grid values to [-1, 1] for resampler
#     for i in range(len(shape)):
#         d= new_locs[:,i, ...]
#         maxd, mind = torch.max(d), torch.min(d)
#         new_locs[:, i, ...] = 2 * (d-mind)/(maxd-mind) -1
#     if len(shape) == 2:
#         new_locs = new_locs.permute(0, 2, 3, 1)
#         new_locs = new_locs[..., [1, 0]]
#     elif len(shape) == 3:
#         new_locs = new_locs.permute(0, 2, 3, 4, 1)
#         new_locs = new_locs[..., [2, 1, 0]]
#     return new_locs
#
# new_locs_works = get_new_locs(moving, works)
# new_locs_dont_work = get_new_locs(moving, does_not_work)
#
# import torch.nn.functional as F
# works_output = F.grid_sample(moving, new_locs_works, align_corners=True)
# does_not_work_output = F.grid_sample(moving, new_locs_dont_work, align_corners=True)
#
#
# moving_warped_with_works = sitk.GetImageFromArray(works_output.numpy().squeeze())
# sitk.WriteImage(moving_warped_with_works, "moving_warped_with_works2.nii.gz")
#
#
# moving_warped_with_not_works = sitk.GetImageFromArray(does_not_work_output.numpy().squeeze())
# sitk.WriteImage(moving_warped_with_not_works, "moving_warped_with_not_works2.nii.gz")
#
# class SpatialTransformer(nn.Module):
#     """N-D Spatial Transformer
#
#     Pulled from VoxelMorph Pytorch implementation
#
#     We also use this module to generate the elastic deformation field
#     as we would like this to happen on the GPU.
#     """
#
#     def __init__(self, size, mode="bilinear"):
#         super().__init__()
#
#         self.mode = mode
#
#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors, indexing="ij")
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.float()
#
#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer("grid", grid)
#
#     def forward(self, src, flow) -> torch.Tensor:
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]
#
#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
