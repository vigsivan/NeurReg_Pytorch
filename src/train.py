"""
Trains the NeurReg model
"""

from tqdm import trange, tqdm
from dataset import ImageDataset
from components import *
from typing import Dict
from torch.nn import Conv3d, Sequential, Softmax, Module, Parameter
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch

from losses import NeurRegLoss
from torch.utils.data import DataLoader

from pathlib import Path
from argparse import ArgumentParser, Namespace

def get_dataloader(params) -> DataLoader:
    dataset = ImageDataset(
        params.imagedir,
        params.segdir,
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
    loss_func = NeurRegLoss(
        params.cross_corr_loss_weight, params.seg_loss_weight, use_cuda=use_cuda
    )
    stn = SpatialTransformer(params.target_shape)

    conv_w_softmax = Sequential(Conv3d(17, 1, 3, padding=1), Softmax(3))

    # Copy strategy from voxelmorph
    to_flow_field = Conv3d(16, 3, 3, padding=1, bias=True)
    to_flow_field.weight = Parameter(Normal(0, 1e-5).sample(to_flow_field.weight.shape))
    if to_flow_field.bias:
        to_flow_field.bias = Parameter(torch.zeros(to_flow_field.bias.shape))

    if 'cuda' in params.device.lower() and params.num_gpus > 1:
        N = torch.nn.DataParallel(N)
        N.state_dict = N.module.state_dict
        # Supposedly speeds up training
        # torch.backends.cudnn.deterministic = True

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
    writer = SummaryWriter(params.logdir)

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


def get_params() -> Namespace:
    help_mssg = ["Trains the Neurreg model.\r"
    "Note, the files in imagedir and segdir should have the same name\r"
    "if they correspond to one another."
    ][0]

    parser = ArgumentParser(description=help_mssg)
    add_arg = parser.add_argument

    add_arg("imagedir", type=Path)
    add_arg("segdir", type=Path)

    add_arg("--target_shape", type=int, nargs="+", default=128, required=False)
    add_arg("--device", type=str, required=False, default="cpu")
    add_arg("--num_gpus", type=int, required=False, default=0)
    add_arg("--num_workers", type=int, required=False, default=4)
    add_arg("--epochs", type=int, required=False, default=1500)
    add_arg("--batch_size", type=int, required=False, default=1)
    add_arg("--lr", type=float, required=False, default=1e-3)
    add_arg("--cross_corr_loss_weight", type=float, required=False, default=10.)
    add_arg("--seg_loss_weight", type=float, required=False, default=10.)

    add_arg("--logdir", type=Path, required=False, default="../logging")
    add_arg("--experiment_name", type=str, required=False, default="experiment1")
    add_arg("--epochs_per_save", type=int, required=False, default=2)

    params = parser.parse_args()
    params.checkpoint = params.experiment_name + "_checkpoint.pt"
    params.step_loss_file = params.experiment_name + "_step_loss.txt"

    if len(params.target_shape) not in (1, 3):
        raise Exception("Target shape should either be 1 or 3")
    if len(params.target_shape) == 3:
        params.target_shape = tuple(params.target_shape)
    else:
        params.target_shape = tuple(params.target_shape*3)
    params.logdir.mkdir(exist_ok=True)

    return params

if __name__ == "__main__":
    params = get_params()
    main(params)
