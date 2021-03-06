"""
Trains the NeurReg model
"""

import sys
from tqdm import trange, tqdm
from dataset import ImageDataset
from components import *
from typing import Dict
from torch.nn import Conv3d, Sequential, Softmax, Module
import params
import torch

from losses import NeurRegLoss
from torch.utils.data import DataLoader

def get_dataloader(params) -> DataLoader:
    dataset = ImageDataset(
        params.path_to_images,
        params.path_to_segs,
        params.matching_fn,
        target_shape=params.target_shape,
    )

    return DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=params.num_workers
    )


def get_models(params) -> Dict[str, Module]:

    N = Unet3D(inshape=params.target_shape)
    loss_func = NeurRegLoss(
        params.cross_corr_loss_weight, params.seg_loss_weight, use_cuda=params.use_cuda
    )
    stn = SpatialTransformer(params.target_shape)

    to_flow_field = Conv3d(16, 3, 3, padding=1)
    conv_w_softmax = Sequential(Conv3d(17, 1, 3, padding=1), Softmax(3))

    if params.use_cuda:
        N = N.cuda()
        to_flow_field = to_flow_field.cuda()
        conv_w_softmax = conv_w_softmax.cuda()
        stn.cuda()

    return {
        "N": N,
        "to_flow_field": to_flow_field,
        "conv_w_softmax": conv_w_softmax,
        "loss_func": loss_func,
        "stn": stn,
    }


def main(params):
    dataloader = get_dataloader(params)
    models = get_models(params)

    N = models["N"]
    to_flow_field = models["to_flow_field"]
    conv_w_softmax = models["conv_w_softmax"]
    loss_func = models["loss_func"]
    stn = models["stn"]

    learnable_params = (
        list(N.parameters())
        + list(to_flow_field.parameters())
        + list(conv_w_softmax.parameters())
    )
    optimizer = torch.optim.Adam(learnable_params, lr=params.lr)

    epochs = params.epochs
    for epoch in trange(epochs):
        for step, data in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            if params.use_cuda:
                for i in ("moving", "another"):
                    for j in ("image", "seg"):
                        data[i][j] = data[i][j].cuda()

                data["transform"] = data["displacement_field"]

            displacement_field = data["transform"]["displacement_field"]
            displacement_field = displacement_field.float()

            transformed_image = stn(data["moving"]["image"], displacement_field)
            transformed_seg = stn(data["moving"]["seg"], displacement_field)

            # Pass images through network as a single batch
            x1 = torch.cat((data["moving"]["image"], transformed_image), dim=1)
            x2 = torch.cat((data["moving"]["image"], data["another"]["image"]), dim=1)

            last_layer = N(x1)
            F_0 = to_flow_field(last_layer)
            F_1 = to_flow_field(N(x2))

            #########################
            # Compute loss. Because we have a lot of variables, we'll use
            # the notation from the paper
            #########################
            F_0g = displacement_field
            I_0 = transformed_image
            I_1 = data["another"]["image"]
            I_0R = stn(data["moving"]["image"], F_0)
            I_1R = stn(data["moving"]["image"], F_1)
            S_0g = transformed_seg
            S_0 = stn(data["moving"]["seg"], F_0)
            S_1 = stn(data["moving"]["seg"], F_1)
            S_1g = stn(data["another"]["seg"], F_1)
            boosted = torch.cat((last_layer, S_0), 1)
            S_0feat = conv_w_softmax(boosted)
            loss = loss_func(F_0, F_0g, I_0, I_0R, I_1, I_1R, S_0feat, S_0g, S_1, S_1g)
            loss.backward()
            optimizer.step()

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
                f.write(f"step={step},loss={loss.item()};")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ("slurm", "cpu"):
        print(f"Usage: {sys.argv[0]} <config_name>\nwhere <config_name> is one of (slurm, cpu)")
        exit(0)
    
    config = sys.argv[1]
    if config == "slurm":
        main(params.SLURM_CONFIG)
    else:
        main(params.CPU_CONFIG)
