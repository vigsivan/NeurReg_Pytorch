"""
Trains the NeurReg model
"""

from math import ceil
from tqdm import trange
from dataset import ImageDataset
from components import *
from typing import Dict
from torch.nn import Conv3d, Sequential, Softmax, Module
import torch.nn.functional as F
import params
import torch

from losses import NeurRegLoss
from torch.utils.data import DataLoader

def get_dataloader() -> DataLoader:
    dataset = ImageDataset(
        params.path_to_images,
        params.path_to_segs,
        params.matching_fn,
        target_shape=params.target_shape,
    )

    return DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=params.num_workers
    )


def get_models() -> Dict[str, Module]:

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

    return {
        "N": N,
        "to_flow_field": to_flow_field,
        "conv_w_softmax": conv_w_softmax,
        "loss_func": loss_func,
        "stn": stn,
    }


def main():
    dataloader = get_dataloader()
    models = get_models()

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
        for step, data in enumerate(dataloader):
            if step == 1:
                break
            optimizer.zero_grad()

            if params.use_cuda:
                for i in ("moving", "another"):
                    for j in ("image", "seg"):
                        data[i][j] = data[i][j].cuda()

                for i in ("affine_field", "elastic_offset", "smoothing_kernel"):
                    data["transform"][i] = data["transform"][i].cuda()
                    data["transform"][i] = data["transform"][i].cuda()
                    data["transform"][i] = data["transform"][i].cuda()

            padding = (data["transform"]["smoothing_kernel"][-1]-1)/2
            elastic_field = F.conv3d(data["transform"]["elastic_offset"].squeeze().unsqueeze(0),
                                     data["transform"]["smoothing_kernel"].squeeze(),
                                     padding = padding)

            displacement_field = elastic_field + data["transform"]["affine_field"]
            transformed_image = stn(data["moving"]["image"], displacement_field)
            transformed_seg = stn(data["moving"]["seg"], displacement_field)

            x1 = torch.cat((data["moving"]["image"], transformed_image), dim=1)
            x2 = torch.cat((data["moving"]["image"], data["another"]["image"]), dim=1)
            batched_images = torch.cat((x1, x2), dim=0)
            last_layer = N(batched_images)
            last_layer0 = last_layer[0, :, :, :, :].unsqueeze(0)
            batched_fields = to_flow_field(last_layer)

            # Using notation from the paper
            F_0g = displacement_field.squeeze().unsqueeze(0)
            F_0 = batched_fields[0, :, :, :, :].unsqueeze(0)
            F_1 = batched_fields[1, :, :, :, :].unsqueeze(0)
            I_0 = transformed_image
            I_1 = data["another"]["image"]
            I_0R = stn(data["moving"]["image"], F_0)
            I_1R = stn(data["moving"]["image"], F_1)
            S_0g = transformed_seg
            S_0 = stn(data["moving"]["seg"], F_0)
            S_1 = stn(data["moving"]["seg"], F_1)
            S_1g = stn(data["another"]["seg"], F_1)
            boosted = torch.cat((last_layer0, S_0), 1)
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
    main()
