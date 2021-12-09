import logging
import os
from pathlib import Path
import random
from typing import List, Tuple, Union

import sys
import SimpleITK as sitk
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
        resize: bool = True,
    ):
        super().__init__()
        self.path_to_images = path_to_images
        self.path_to_segmentations = path_to_segmentations

        self.stn = SpatialTransformer(target_shape)

        if resize:
            self.size_fn = tio.Resize(target_shape)
            logging.info("Resizing images")
        else:
            self.size_fn = tio.CropOrPad(target_shape, padding_mode=0)
            logging.info("Cropping/padding images")

        self.rescale = tio.RescaleIntensity()

        self.images: List[Union[Path, torch.Tensor]] = [
            i for i in self.files_generator(path_to_images)
        ]
        self.segs: List[Union[Path, torch.Tensor]] = [
            i for i in self.files_generator(path_to_segmentations)
        ]

        reference_tio = tio.ScalarImage(str(self.images[0]))
        reference = self.size_fn(reference_tio).as_sitk()
        self.simulator = RegistrationSimulator3D(reference, target_shape)

        self.data_consistency()

    def files_generator(self, dir: Path):
        dir_files = [
            dir / i for i in os.listdir(dir) if i.endswith(".nii.gz") and i[0] != "."
        ]
        dir_files.sort()
        yield from dir_files

    def data_consistency(self) -> None:
        if len(self.images) != len(self.segs):
            nim, nse = len(self.images), len(self.segs)
            raise Exception(f"Number of images and segs don't match: {nim} vs {nse}")
        for image, seg in zip(self.images, self.segs):
            if isinstance(image, Path) and isinstance(seg, Path):
                sim, sse = image.name, seg.name
                if sim != sse:
                    raise Exception(
                        f"Image file and seg file don't match: {sim} vs {sse}"
                    )

    def __len__(self):
        return len(self.images)

    def process(
        self, x: Union[Path, torch.Tensor], index: int, is_seg: bool = False
    ) -> torch.Tensor:
        """
        Our strategy is to cache the processed images so we need an early-exit
        if already cached.
        """
        if isinstance(x, torch.Tensor):
            return x
        x = str(x)
        x_tio = tio.LabelMap(x) if is_seg else tio.ScalarImage(x)
        x_tensor = x_tio.data.T.squeeze().unsqueeze(0)
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

        transform_field = self.simulator.get_displacement_tensor().float()
        transformed_image: torch.Tensor = (
            self.stn(moving_image.unsqueeze(0), transform_field).squeeze().unsqueeze(0)
        )
        transformed_seg: torch.Tensor = (
            self.stn(moving_seg.unsqueeze(0), transform_field).squeeze().unsqueeze(0)
        )
        transformed_image = transformed_image
        transformed_seg = transformed_seg.squeeze().unsqueeze(0)

        return (
            moving_image,
            moving_seg,
            target_image,
            target_seg,
            transform_field,
            transformed_image,
            transformed_seg,
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path_to_images> <path_to_segmentations>")
    p2ims, p2segs = Path(sys.argv[1]), Path(sys.argv[2])
    dataset = ImageDataset(p2ims, p2segs, target_shape=(128, 128, 128), resize=True)
    index = random.randint(0, len(dataset) - 1)
    outs = dataset[index]
    transformed = sitk.GetImageFromArray(outs[-2].numpy())
    transform = sitk.GetImageFromArray(outs[4].numpy())
    moving = sitk.GetImageFromArray(outs[0].squeeze().numpy())
    sitk.WriteImage(moving, "moving.nii.gz")
    sitk.WriteImage(transformed, "transformed.nii.gz")
    sitk.WriteImage(transform, "transform.nii.gz")
