import os
import random
import torchio as tio
import torch
from params import CPU_CONFIG as params
from typing import Callable, Dict, Optional, Tuple
from pathlib import Path

from components import *
from torch.utils.data import Dataset, DataLoader


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
        super().__init__()
        self.path_to_images = path_to_images
        self.path_to_segmentations = path_to_segmentations
        self.pad_fn = tio.CropOrPad(target_shape, padding_mode=0)

        self.registration_simulator = (
            registration_simulator
            if registration_simulator
            else RegistrationSimulator3D()
        )

        self.stn = SpatialTransformer(target_shape)

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

    def __getitem__(self, index: int):
        data: Dict[str, Dict[str, torch.Tensor]] = {
            "moving": {},
            "another": {},
            "transform": {},
        }

        pad = lambda x: self.pad_fn(x.squeeze().unsqueeze(0)).float()

        moving_image_file, moving_seg_file = self.images[index], self.segs[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        moving_seg_tio = tio.LabelMap(self.path_to_segmentations / moving_seg_file)

        randindex = random.randint(0, len(self.images) - 1)
        image_file, seg_file = self.images[randindex], self.segs[randindex]

        # Expected format of the images is in B,C,D,W,H format
        # the pad function does all of the squeezing/unsqueezing necessary for 3D inputs
        moving_image = pad(moving_image_tio.data)
        moving_seg = pad(moving_seg_tio.data)
        another_image = pad(tio.ScalarImage(self.path_to_images / image_file).data)
        another_seg = pad(tio.LabelMap(self.path_to_segmentations / seg_file).data)

        displacement_field = self.registration_simulator(
            tio.ScalarImage(tensor=moving_image)
        ).float()

        transform_image = (
            self.stn(moving_image.unsqueeze(0), displacement_field.unsqueeze(0))
            .squeeze()
            .unsqueeze(0)
        )

        transform_seg = (
            self.stn(moving_seg.unsqueeze(0), displacement_field.unsqueeze(0))
            .squeeze()
            .unsqueeze(0)
        )

        concat1 = torch.cat((moving_image, transform_image), dim=0)
        concat2 = torch.cat((moving_image, another_image), dim=0)

        data["transform"]["concat"] = concat1
        data["another"]["concat"] = concat2

        data["moving"]["image"] = moving_image
        data["moving"]["seg"] = moving_seg

        data["another"]["image"] = another_image
        data["another"]["seg"] = another_seg

        data["transform"]["image"] = transform_image
        data["transform"]["seg"] = transform_seg
        data["transform"]["field"] = displacement_field

        return data


if __name__ == "__main__":
    dataset = ImageDataset(
        params.path_to_images,
        params.path_to_segs,
        params.matching_fn,
        target_shape=params.target_shape,
    )
    import random

    data = random.choice(dataset)
    # dataloader = DataLoader(dataset, batch_size=6)
    # data = next(iter(dataloader))
