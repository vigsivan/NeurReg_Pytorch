"""
Trains the NeurReg model
"""

import sys
from tqdm import trange, tqdm
from dataset import ImageDataset
from components import *
from typing import Dict
from torch.nn import Conv3d, Sequential, Softmax, Module, Parameter
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import params
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

    dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

    use_cuda = "cuda" in params.device
    if use_cuda:
        dl.pin_memory = True

    return dl


def get_models(params) -> Dict[str, Module]:

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

    if 'cuda' in params.device and params.num_gpus > 1:
        N = torch.nn.DataParallel(N)
        N.state_dict = N.module.state_dict
        # Supposedly speeds up training
        torch.backends.cudnn.deterministic = True

    conv_w_softmax.train()
    N.train()

    return {
        "N": N,
        "to_flow_field": to_flow_field,
        "conv_w_softmax": conv_w_softmax,
        "stn": stn,
        "loss_func": loss_func,
    }

def main(params):
    dataloader = get_dataloader(params)
    models = get_models(params)
    writer = SummaryWriter(params.savedir)

    N = models["N"].to(params.device)
    to_flow_field = models["to_flow_field"].to(params.device)
    conv_w_softmax = models["conv_w_softmax"].to(params.device)
    loss_func = models["loss_func"]
    stn = models["stn"].to(params.device)

    learnable_params = (
        list(N.parameters())
        + list(to_flow_field.parameters())
        + list(conv_w_softmax.parameters())
    )
    optimizer = torch.optim.Adam(learnable_params, lr=params.lr)

    epochs = params.epochs
    total_steps = 0
    for epoch in trange(epochs):
        for _, data in tqdm(enumerate(dataloader)):

            for category in ("moving", "another", "transform"):
                for tensor in ("image", "seg"):
                    data[category][tensor] = data[category][tensor].to(params.device)
                if category == "transform":
                    data[category]["field"] = data[category]["field"].to(params.device)
                    data[category]["concat"] = data[category]["concat"].to(
                        params.device
                    )
                if category == "another":
                    data[category]["concat"] = data[category]["concat"].to(
                        params.device
                    )

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
            S_1g = stn(data["another"]["seg"], F_1)
            boosted = torch.cat((last_layer, S_0), 1)
            S_0feat = conv_w_softmax(boosted)
            loss = loss_func(F_0, F_0g, I_0, I_0R, I_1, I_1R, S_0feat, S_0g, S_1, S_1g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

        writer.add_scalar("Loss/train", loss.item())

        if epoch % params.epochs_per_save == 0:
            torch.save(
                {
                    "N": N.state_dict(),
                    "to_flow_field": to_flow_field.state_dict(),
                    "conv_w_softmax": conv_w_softmax.state_dict(),
                },
                str(params.checkpoint),
            )
            with open(params.step_loss_file, "a") as f:
                f.write(f"step={total_steps},loss={loss.item()};")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ("slurm", "cpu"):
        print( f"Usage: {sys.argv[0]} <config_name>\nwhere <config_name> is one of (slurm, cpu)"
        )
        exit(0)

    config = sys.argv[1]
    if config == "slurm":
        print("Using SLURM CONFIG")
        main(params.SLURM_CONFIG)
    else:
        print("Using CPU CONFIG")
        main(params.CPU_CONFIG)
