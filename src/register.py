"""
Registers using a trained NeurReg model
"""

from components import *
from typing import Tuple
from torch.nn import Conv3d, Sequential, Softmax, Module
import torch

from pathlib import Path
from argparse import ArgumentParser, Namespace
import torchio as tio


def load_models(params) -> Tuple[Module, Module, Module, Module]:

    N = Unet3D(inshape=params.target_shape)
    stn = SpatialTransformer(params.target_shape)

    conv_w_softmax = Sequential(Conv3d(17, 1, 3, padding=1), Softmax(3))

    # Copy strategy from voxelmorph
    to_flow_field = Conv3d(16, 3, 3, padding=1, bias=True)

    if params.device.lower() == "cpu":
        checkpoint = torch.load(params.checkpoint, map_location=torch.device("cpu"))
    else:
        N = N.to(params.device)
        to_flow_field = to_flow_field.to(params.device)
        conv_w_softmax = conv_w_softmax.to(params.device)
        stn = stn.to(params.device)

        checkpoint = torch.load(params.checkpoint)

    N.load_state_dict(checkpoint["N"])
    to_flow_field.load_state_dict(checkpoint["to_flow_field"])
    conv_w_softmax.load_state_dict(checkpoint["conv_w_softmax"])

    to_flow_field.eval()
    conv_w_softmax.eval()
    N.eval()

    return N, to_flow_field, conv_w_softmax, stn


def process_data(params) -> None:
    rescale = tio.RescaleIntensity()
    if params.shape_op == "pad":
        tsfm = tio.Compose([tio.Resize(params.target_shape), rescale])
    else:
        tsfm = tio.Compose([tio.CropOrPad(params.target_shape, padding=0), rescale])

    params.fixed = tsfm(tio.ScalarImage(params.fixed)).data.float().unsqueeze(0)
    params.moving = tsfm(tio.ScalarImage(params.moving)).data.float().unsqueeze(0)
    if params.moving_seg:
        params.moving_seg = (
            tsfm(tio.LabelMap(params.moving_seg)).data.float().unsqueeze(0)
        )


def get_params() -> Namespace:
    help_mssg = "Registers with a trained Neurreg model."

    parser = ArgumentParser(description=help_mssg)
    add_arg = parser.add_argument

    add_arg("moving", type=Path)
    add_arg("fixed", type=Path)
    add_arg("checkpoint", type=Path)

    add_arg("--output_file", type=Path, default=Path("./moved.nii.gz"), required=False)
    add_arg("--moving_seg", type=Path, required=False, default=None)
    add_arg(
        "--output_seg_file",
        type=Path,
        required=False,
        default=Path("./output_seg.nii.gz"),
    )
    add_arg("--target_shape", type=int, nargs="+", default=[128], required=False)
    add_arg(
        "--shape_op", type=str, choices=("resize", "pad"), required=False, default="pad"
    )
    add_arg("--save_fields", type=bool, required=False, default=False)
    add_arg("--device", type=str, required=False, default="cpu")

    params = parser.parse_args()

    if len(params.target_shape) not in (1, 3):
        raise Exception("Target shape should either be 1 or 3")
    if len(params.target_shape) == 3:
        params.target_shape = tuple(params.target_shape)
    else:
        params.target_shape = tuple(params.target_shape * 3)

    process_data(params)

    return params


def main(params):
    N, to_flow_field, conv_w_softmax, stn = load_models(params)

    concat = torch.concat([params.moving, params.fixed], dim=1).to(params.device)

    last_layer = N(concat)
    F = to_flow_field(last_layer)

    moved_image = stn(params.moving, F).squeeze().unsqueeze(0).detach()
    tio.ScalarImage(tensor=moved_image).save(params.output_file)

    if params.moving_seg:
        moved_seg = stn(params.moving_seg, F)
        boosted = torch.cat((last_layer, moved_seg), 1)
        moved_seg_boosted = conv_w_softmax(boosted)
        tio.LabelMap(tensor=moved_seg_boosted.squeeze()).save(params.output_seg_file)


if __name__ == "__main__":
    params = get_params()
    main(params)
