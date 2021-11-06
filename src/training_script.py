"""
Trains the NeurReg model
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List

import torch
import torchio as tio
from components import *

# from losses import NeurRegLoss
from torch.utils.data import random_split, Dataset


class ImageDataset(Dataset):
    """ """

    def __init__(
        self,
        path_to_images: Path,
        path_to_segmentations: Path,
        matching_fn: Callable,
        target_shape: Tuple[int, int, int],
        registration_simulator: Optional[RegistrationSimulator3D] = None,
    ):
        self.path_to_images = path_to_images
        self.path_to_segmentations = path_to_segmentations
        self.pad_fn = tio.CropOrPad(target_shape, padding_mode=0)

        self.registration_simulator = (
            registration_simulator
            if registration_simulator
            else RegistrationSimulator3D()
        )

        images = os.listdir(path_to_images)
        segmentations = os.listdir(path_to_segmentations)

        self.images, self.segs = self.match(images, segmentations, matching_fn)
        assert len(self.images) == len(self.segs)

    def match(self, prefix_list, added_list, matching_fn):
        sorted_prefix_list, sorted_added_list = [], []
        for i in prefix_list:
            if matching_fn(i) in added_list:
                sorted_prefix_list.append(i)
                sorted_added_list.append(matching_fn(i))
        return sorted_prefix_list, sorted_added_list

    def __len__(self):
        return len(self.images)

    def __get_deformation_field(self, index: int):
        moving_image_file = self.images[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        deformation_field = self.registration_simulator(moving_image_tio).data
        return deformation_field.unsqueeze(0).float()

    def __get_spatial_transformer(self, index: int, deformation_field: torch.Tensor):
        moving_image_file = self.images[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        stn = SpatialTransformer(moving_image_tio.data.squeeze().shape)

        tsfm = lambda x: stn(
            x.squeeze().unsqueeze(0).unsqueeze(0).float(), deformation_field
        )
        return tsfm

    def __pad(self, image: torch.Tensor) -> torch.Tensor:
        padded = self.pad_fn(image.squeeze().unsqueeze(0)).unsqueeze(0).float()
        return padded

    def __getitem__(self, index: int):
        data: Dict[str, Dict[str, torch.Tensor]] = {
            "moving": {},
            "another": {},
            "transformed": {},
        }

        moving_image_file, moving_seg_file = self.images[index], self.segs[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        moving_seg_tio = tio.LabelMap(self.path_to_segmentations / moving_seg_file)

        randindex = random.randint(0, len(self.images) - 1)
        image_file, seg_file = self.images[randindex], self.segs[randindex]

        deformation_field = self.__get_deformation_field(index)
        stn = self.__get_spatial_transformer(index, deformation_field)

        # Expected format of the images is in B,C,D,W,H format
        # the pad function does all of the squeezing/unsqueezing necessary for 3D inputs
        moving_image = self.__pad(moving_image_tio.data)
        moving_seg = self.__pad(moving_seg_tio.data)
        transformed_image = self.__pad(stn(moving_image))
        transformed_seg = self.__pad(stn(moving_seg))
        another_image = self.__pad(
            tio.ScalarImage(self.path_to_images / image_file).data
        )
        another_seg = self.__pad(
            tio.LabelMap(self.path_to_segmentations / seg_file).data
        )

        data["moving"]["image"] = moving_image
        data["moving"]["seg"] = moving_seg

        data["transformed"]["image"] = transformed_image
        data["transformed"]["seg"] = transformed_seg

        data["another"]["image"] = another_image
        data["another"]["seg"] = another_seg

        return data


path_to_images = "/Volumes/Untitled/Task04_Hippocampus/imagesTr/"
path_to_segs = "/Volumes/Untitled/Task04_Hippocampus/labelsTr/"
matching_fn = lambda x: x

dataset = ImageDataset(
    Path(path_to_images), Path(path_to_segs), matching_fn, target_shape=(128, 128, 128)
)

train_proportion = 0.9
train_len = int(train_proportion * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# network = RegistrationNetwork3D()
data = dataset[random.randint(0, len(dataset) - 1)]

for _ in range(20):
    data = dataset[random.randint(0, len(dataset) - 1)]

x1 = torch.cat((data["moving"]["image"], data["transformed"]["image"]), dim=1)
x2 = torch.cat((data["moving"]["image"], data["another"]["image"]), dim=1)
batched_input = torch.cat((x1, x2), dim=0)

# dims: int = 3
# lrelu_slope: float = 0.2
# kernel_size: int = 3
# encoder_channels: List[int] = [16, 32, 32, 32]
# decoder_channels: List[int] = [32, 32, 32, 16]
# bottleneck_channels: List[int] = [32, 32]
# encoder_stride: int = 2
# encoder_channels = [2, *encoder_channels]
#
# from torch.nn import Sequential, ModuleList, Conv3d, ConvTranspose3d
#
# encoder_layers = ModuleList(
#     [
#         Conv3d(
#             in_channels=ic,
#             out_channels=oc,
#             kernel_size=kernel_size,
#             stride=1,
#         )
#         for ic, oc in zip(encoder_channels, encoder_channels[1:])
#     ]
# )
#
# encoder_inputs = [batched_input]
# for encoder_layer in encoder_layers:
#     encoder_inputs.append(encoder_layer(encoder_inputs[-1]))
#
# bottleneck_channels = [encoder_channels[-1], *bottleneck_channels]
# bottleneck = Sequential(
#     *[
#         Conv3d(in_channels=ic, out_channels=oc, kernel_size=kernel_size)
#         for ic, oc in zip(bottleneck_channels, bottleneck_channels[1:])
#     ]
# )
#
# bottleneck_inputs = [encoder_inputs[-1]]
# for bl in bottleneck:
#     bottleneck_inputs.append(bl(bottleneck_inputs[-1]))
#
# upsampling_layers = ModuleList(
#     [
#         ConvTranspose3d(in_channels=ic, out_channels=ic, kernel_size=kernel_size)
#         for ic in [bottleneck_channels[-1], *decoder_channels[:-1]]
#     ]
# )

# decoder_channels = [decoder_channels[0], *decoder_channels]
# decoder_conv_layers = ModuleList(
#     [
#         Conv3d(in_channels=ic, out_channels=oc, kernel_size=kernel_size)
#         for ic, oc in zip(decoder_channels, decoder_channels[1:])
#     ]
# )
#
# output_conv_layer = Conv3d(
#     in_channels=decoder_channels[-1], out_channels=dims, kernel_size=3
# )
#
# M, S_M = [i for i in data["moving"].values()]
# I0, S_0 = [i for i in data["transformed"].values()]
# x = torch.cat((M, I0),1)
# encoder_outs = [x]
# for encoder in encoder_layers:
#     encoder_outs.append(encoder(encoder_outs[-1]))
#
# encoder_outs = [encoder_layers[0](x)]
# for i in range(1, len(encoder_layers)):
#     x_act = F.leaky_relu(encoder_outs[-1], lrelu_slope)
#     encoder_outs.append(encoder_layers[i](x_act))
# decoder_input = bottleneck(encoder_outs[-1])
# for eout, upsample, decoder_conv in zip(
#     encoder_outs, upsampling_layers, decoder_conv_layers
# ):
#     out = upsample(decoder_input)
#     out = torch.cat(eout, out)
#     out = decoder_conv(out)
#     decoder_input = F.leaky_relu(out, lrelu_slope)
#
# F_N = decoder_input  # final feature layer
# output = output_conv_layer(decoder_input)
