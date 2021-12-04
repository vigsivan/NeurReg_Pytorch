"""
Trains the NeurReg model
"""

from tqdm import trange, tqdm
from dataset import ImageDataset
from components import *
from torch.utils.tensorboard import SummaryWriter
import logging
import torch

from models import NeurRegNet

from losses import (
    tversky_loss2,
    registration_field_loss,
    local_cross_correlation_loss3D,
)

from torch.utils.data import DataLoader

from pathlib import Path
from argparse import ArgumentParser, Namespace


def get_dataloader(params) -> DataLoader:
    dataset = ImageDataset(
        params.imagedir,
        params.segdir,
        target_shape=params.target_shape,
        transform=params.transform_cpu,
        resize=params.shape_op == "resize",
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


def main(params):
    dataloader = get_dataloader(params)
    writer = SummaryWriter(params.logdir)

    simulator = RegistrationSimulator3D()
    model = NeurRegNet(params.target_shape, transform_inputs=not params.transform_cpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    tsum = lambda t: t[0] + t[1]

    epochs = params.epochs
    total_steps = 0
    for epoch in trange(epochs):
        steps = 0
        epoch_loss = 0
        for _, data in tqdm(enumerate(dataloader)):
            steps += 1
            optimizer.zero_grad()

            if not params.transform_cpu:
                (mov_im, mov_seg, targ_im, targ_seg) = data
                transform = simulator(mov_im)
                outputs = model(mov_im, mov_seg, targ_im, transform)

            else:
                (mov_im, mov_seg, targ_im, targ_seg,
                transform, transformed_image, transformed_seg) = data

                outputs = model(mov_im, 
                                mov_seg, 
                                targ_im, 
                                transform, 
                                transformed_image,
                                transformed_seg)

            field_pair = (transform, outputs.moving_to_precomputed_field)
            image_pairs = (
                (outputs.precomputed_image, outputs.moving_to_precomputed_image),
                (targ_im, outputs.moving_to_target_image),
            )
            seg_pairs = (
                (
                    outputs.precomputed_segmentation,
                    outputs.moving_to_precomputed_segmentation,
                ),
                (targ_seg, outputs.moving_to_target_segmentation),
            )

            l_image = lambda p: local_cross_correlation_loss3D(
                *p, use_cuda="cuda" in params.device
            )

            field_loss = registration_field_loss(*field_pair)
            image_loss = tsum([l_image(p) for p in image_pairs])
            seg_loss = tsum([tversky_loss2(*p) for p in seg_pairs])

            loss = (
                field_loss
                + params.cross_corr_loss_weight * image_loss
                + params.seg_loss_weight * seg_loss
            )

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            total_steps += 1

        writer.add_scalar(f"Average loss", epoch_loss / steps)

        if epoch % params.epochs_per_save == 0:
            torch.save(model.state_dict(), str(params.checkpoint))
            with open(params.step_loss_file, "a") as f:
                f.write(f"step={total_steps},loss={loss.item()};")


def get_params() -> Namespace:
    help_mssg = [
        "Trains the Neurreg model.\r"
        "Note, the files in imagedir and segdir should have the same name\r"
        "if they correspond to one another."
    ][0]

    parser = ArgumentParser(description=help_mssg)
    add_arg = parser.add_argument

    add_arg("imagedir", type=Path)
    add_arg("segdir", type=Path)

    add_arg("--target_shape", type=int, nargs="+", default=(128), required=False)
    add_arg(
        "--shape_op", type=str, choices=("resize", "pad"), required=False, default="pad"
    )
    add_arg("--device", type=str, required=False, default="cpu")
    add_arg("--num_gpus", type=int, required=False, default=0)
    add_arg("--num_workers", type=int, required=False, default=4)
    add_arg("--epochs", type=int, required=False, default=1500)
    add_arg("--batch_size", type=int, required=False, default=1)
    add_arg("--lr", type=float, required=False, default=1e-3)
    add_arg("--cross_corr_loss_weight", type=float, required=False, default=10.0)
    add_arg("--seg_loss_weight", type=float, required=False, default=10.0)

    add_arg("--logdir", type=Path, required=False, default="../logging")
    add_arg("--experiment_name", type=str, required=False, default="experiment1")
    add_arg("--epochs_per_save", type=int, required=False, default=2)
    add_arg("--transform-cpu", help="Compute transforms on the cpu", action="store_true")

    params = parser.parse_args()
    params.checkpoint = params.experiment_name + "_checkpoint.pt"
    params.step_loss_file = params.experiment_name + "_step_loss.txt"

    if len(params.target_shape) not in (1, 3):
        raise Exception("Target shape should either be 1 or 3")
    if len(params.target_shape) == 3:
        params.target_shape = tuple(params.target_shape)
    else:
        params.target_shape = tuple(params.target_shape * 3)
    params.logdir.mkdir(exist_ok=True)

    return params


if __name__ == "__main__":
    params = get_params()
    logfile = params.logdir / (params.experiment_name + ".log")
    logging.basicConfig(filename=str(logfile), level=logging.INFO)
    main(params)
