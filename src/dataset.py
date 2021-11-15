import os
import sys
import random
import torchio as tio
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path

from components import *
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Image Dataset.

    Note, corresponding images and segmentations need to have the same file name.

    Parameters
    ----------
    path_to_images: Path
    path_to_segmentations: Path
    target_shape: Tuple[int,int,int]
    registration_simulator: Optional[RegistrationSimulator3D]
        default=None
    """

    def __init__(
        self,
        path_to_images: Path,
        path_to_segmentations: Path,
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

        self.rescale = tio.RescaleIntensity()

        self.stn = SpatialTransformer(target_shape)

        self.images = [i for i in self.dir_generator(path_to_images)]
        self.segs = [i for i in self.dir_generator(path_to_segmentations)]

        self.images.sort()
        self.segs.sort()

        try:
            assert self.data_consistency()
        except AssertionError:
            breakpoint()
            print("Length images: ", len(self.images)) 
            print("Length segs: ", len(self.segs)) 

    def dir_generator(self, dir: Path):
        for i in os.listdir(dir):
            if i.endswith(".nii.gz") and i[0] != ".":
                yield i

    def data_consistency(self) -> bool:
        if len(self.images) != len(self.segs):
            return False
        for i, s in zip(self.images, self.segs):
            if i != s:
                return False
        return True

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        data: Dict[str, Dict[str, torch.Tensor]] = {
            "moving": {},
            "another": {},
            "transform": {},
        }

        process = lambda x: self.rescale(self.pad_fn(x.squeeze().unsqueeze(0))).float()

        moving_image_file, moving_seg_file = self.images[index], self.segs[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        moving_seg_tio = tio.LabelMap(self.path_to_segmentations / moving_seg_file)

        randindex = random.randint(0, len(self.images) - 1)
        image_file, seg_file = self.images[randindex], self.segs[randindex]

        # Expected format of the images is in B,C,D,W,H format
        # the pad function does all of the squeezing/unsqueezing necessary for 3D inputs
        moving_image = process(moving_image_tio.data)
        moving_seg = process(moving_seg_tio.data)
        another_image = process(tio.ScalarImage(self.path_to_images / image_file).data)
        another_seg = process(tio.LabelMap(self.path_to_segmentations / seg_file).data)

        displacement_field = self.registration_simulator(
            tio.ScalarImage(tensor=moving_image)
        ).float()

        transform_image = self.rescale(
            (
                self.stn(moving_image.unsqueeze(0), displacement_field.unsqueeze(0))
                .squeeze()
                .unsqueeze(0)
            )
        )

        transform_seg = self.rescale(
            (
                self.stn(moving_seg.unsqueeze(0), displacement_field.unsqueeze(0))
                .squeeze()
                .unsqueeze(0)
            )
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
    if len(sys.argv) != 4:
        print("Usage: sys.argv[0] <imagedir> <segdir> <target_shape_int>")
        exit("0")

    path_to_images, path_to_segs, target_shape = (
        sys.argv[1],
        sys.argv[2],
        tuple([int(sys.argv[3])] * 3),
    )
    dataset = ImageDataset(
        Path(path_to_images),
        Path(path_to_segs),
        target_shape=target_shape,
    )
    import random

    data = random.choice(dataset)
