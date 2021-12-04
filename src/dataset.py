import logging
import os
from pathlib import Path
import random
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio

from components import *


class ImageDataset(Dataset):
    """
    Image Generator.

    Note, corresponding images and segmentations need to have the same file name.

    Parameters
    ----------
        self,
        path_to_images: Path,
        path_to_segmentations: Path,
        target_shape: Tuple[int, int, int],
        transform: bool
            If True, the image is transformed and transformed images are returned
        resize: bool
            If True, the images are resized (through interpolation).
            Else, they are cropped/padded.
            Default=True.
    """

    def __init__(
        self,
        path_to_images: Path,
        path_to_segmentations: Path,
        target_shape: Tuple[int, int, int],
        transform: bool=True,
        resize: bool = False,
    ):
        super().__init__()
        self.path_to_images = path_to_images
        self.path_to_segmentations = path_to_segmentations

        self.transform = transform
        self.simulator = RegistrationSimulator3D()
        self.stn = SpatialTransformer(target_shape)

        if resize:
            self.size_fn = tio.Resize(target_shape)
            logging.info("Resizing images")
        else:
            self.size_fn = tio.CropOrPad(target_shape, padding_mode=0)
            logging.info("Cropping/padding images")

        self.rescale = tio.RescaleIntensity()

        self.images: List[Union[str, torch.Tensor]] = [
            i for i in self.files_generator(path_to_images)
        ]
        self.segs: List[Union[str, torch.Tensor]] = [
            i for i in self.files_generator(path_to_segmentations)
        ]

        self.data_consistency()

    def files_generator(self, dir: Path):
        dir_files = [
            i
            for i in os.listdir(dir)
            if i.endswith(".nii.gz") and i[0] != "." and "sub" in i
        ]
        dir_files.sort()
        yield from dir_files

    def data_consistency(self) -> None:
        if len(self.images) != len(self.segs):
            nim, nse = len(self.images), len(self.segs)
            raise Exception(f"Number of images and segs don't match: {nim} vs {nse}")
        for i, s in zip(self.images, self.segs):
            if isinstance(i, str) and isinstance(s, str) and i != s:
                sim, sse = str(i), str(s)
                raise Exception(f"Image file and seg file don't match: {sim} vs {sse}")

    def __iter__(self):
        indices = np.arange(0, len(self))
        np.random.shuffle(indices)
        self.images = [self.images[i] for i in indices]
        self.segs = [self.segs[i] for i in indices]
        return self

    def __len__(self):
        return len(self.images)

    def process(
        self, x: Union[str, torch.Tensor], index: int, is_seg: bool = False
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if is_seg:
            x_tio = tio.LabelMap(self.path_to_segmentations / x)
        else:
            x_tio = tio.ScalarImage(self.path_to_images / x)

        x_tensor = x_tio.data.squeeze().unsqueeze(0)
        processed = torch.Tensor(self.rescale(self.size_fn(x_tensor))).float()
        if is_seg:
            self.segs[index] = processed
        else:
            self.images[index] = processed

        return processed

    def __getitem__(self, index: int):
        moving_image = self.process(self.images[index], index)
        moving_seg = self.process(self.segs[index], index, is_seg=True)

        next_index = random.randint(0, len(self) - 1)
        target_image = self.process(self.images[next_index], index)
        target_seg = self.process(self.segs[next_index], index, is_seg=True)

        if self.transform:
            transform_field: torch.Tensor = self.simulator(moving_image)
            transformed_image: torch.Tensor = self.stn(moving_image, transform_field)
            transformed_seg: torch.Tensor = self.stn(moving_seg, transform_field)

            # remove batched dimension
            transformed_image = transformed_image.squeeze().unsqueeze(0)
            transformed_seg = transformed_seg.squeeze().unsqueeze(0)

            return (moving_image, moving_seg, target_image, target_seg, transform_field, transformed_image, transformed_seg)

        return moving_image, moving_seg, target_image, target_seg


if __name__ == "__main__":
    pass
