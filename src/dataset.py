import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio

from components import *


class ImageDataset(Dataset):
    """
    Image Generator.

    Note, corresponding images and segmentations need to have the same file name.

    Parameters
    ----------
    path_to_images: Path
    path_to_segmentations: Path
    target_shape: Tuple[int,int,int]
    prob_same: float
        A number between 0 and 1 for the probability that the two numbers
    resize: bool
        If True, this resizes the image. Else, the image is cropped or padded to target
    registration_simulator: Optional[RegistrationSimulator3D]
        default=None
        If False, does not return transformed images
    """

    def __init__(
        self,
        path_to_images: Path,
        path_to_segmentations: Path,
        target_shape: Tuple[int, int, int],
        prob_same: float = 0.0,
        resize: bool = False,
        registration_simulator: Optional[RegistrationSimulator3D] = None,
    ):
        super().__init__()
        self.path_to_images = path_to_images
        self.path_to_segmentations = path_to_segmentations
        self.prob_same = prob_same
        if resize:
            self.size_fn = tio.Resize(target_shape)
            logging.info("Resizing images")
        else:
            self.size_fn = tio.CropOrPad(target_shape, padding_mode=0)
            logging.info("Cropping/padding images")

        self.registration_simulator = registration_simulator

        self.rescale = tio.RescaleIntensity()
        self.stn = (
            SpatialTransformer(target_shape) if self.registration_simulator else None
        )

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
        # NOTE: the returned tensor should have shape 1, 1, *target_shape
        # for this reason, we unsqueeze here
        # return processed.unsqueeze(0)

    def __getitem__(self, index: int):
        data: Dict[str, Dict[str, torch.Tensor]] = {
            "moving": {},
            "another": {},
        }

        moving_image = self.process(self.images[index], index)
        moving_seg = self.process(self.segs[index], index, is_seg=True)

        next_index = (index + 1) % len(self)
        # next_index = index if random.random() < self.prob_same else random.randint(0, len(self)-1)
        another_image = self.process(self.images[next_index], index)
        another_seg = self.process(self.segs[next_index], index, is_seg=True)

        data["moving"]["image"] = moving_image
        data["moving"]["seg"] = moving_seg
        data["another"]["image"] = another_image
        data["another"]["seg"] = another_seg
        # data["another"]["concat"] = torch.cat((moving_image, another_image), dim=0)

        if self.registration_simulator is not None and self.stn is not None:
            data["transform"] = {}

            transform = self.registration_simulator(
                tio.ScalarImage(tensor=moving_image)
            )
            # NOTE: Transform needs to have shape 1, 3, *target_shape
            transform = transform.squeeze().unsqueeze(0)
            transformed_image = self.stn(moving_image, transform)
            transformed_seg = self.stn(moving_seg, transform)
            # concat_transform = torch.cat((moving_image, transformed_image), dim=0)

            data["transform"]["image"] = transformed_image
            data["transform"]["seg"] = transformed_seg
            data["transform"]["field"] = transform
            # data["transform"]["concat"] = concat_transform

        return data


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <imagedir> <segdir> <target_shape_int> <action>")
        exit("0")

    shape = int(sys.argv[3])
    tshape = (shape, shape, shape)
    path_to_images, path_to_segs, target_shape = (
        sys.argv[1],
        sys.argv[2],
        tshape,
    )
    action = "time" if len(sys.argv) < 5 else sys.argv[4]
    if action not in ("time", "dataloader"):
        raise ValueError("Incorrect parameter for value")
    if action == "time":

        # print("*"*50 + "\nWithout transforms")
        # dataset = ImageDataset(
        #     Path(path_to_images),
        #     Path(path_to_segs),
        #     target_shape=target_shape,
        #     resize=False,
        # )
        #
        # import timeit
        # time_init = timeit.timeit(lambda: [dataset[i] for i in range(len(dataset))], number=1)
        # time_cache = timeit.timeit(lambda: [dataset[i] for i in range(len(dataset))], number=1)
        # print("Init time: ", time_init)
        # print("Cache time: ", time_cache)

        print("*" * 50 + "\nWith transforms")
        dataset = ImageDataset(
            Path(path_to_images),
            Path(path_to_segs),
            target_shape=target_shape,
            registration_simulator=RegistrationSimulator3D(),
            resize=False,
        )

        import timeit

        time_init = timeit.timeit(
            lambda: [dataset[i] for i in range(len(dataset))], number=1
        )
        time_cache = timeit.timeit(
            lambda: [dataset[i] for i in range(len(dataset))], number=1
        )
        print("Init time: ", time_init)
        print("Cache time: ", time_cache)

    if action == "dataloader":
        dataset = ImageDataset(
            Path(path_to_images),
            Path(path_to_segs),
            target_shape=target_shape,
            resize=False,
        )

        dataloader = DataLoader(dataset, shuffle=True)
        for _ in dataloader:
            pass
        print("Completed a single epoch through the dataset with a dataloader")
