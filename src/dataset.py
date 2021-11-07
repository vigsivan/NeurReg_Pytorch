import os
import random
import torchio as tio
import torch
from typing import Callable, Dict, Optional, Tuple, Union
from pathlib import Path

from components import *
from torch.utils.data import Dataset

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

    def __get_deformation_field(self, image: tio.ScalarImage):
        deformation_field = self.registration_simulator(image).data
        return deformation_field.unsqueeze(0).float()


    def __getitem__(self, index: int):
        data: Dict[str, Dict[str, torch.Tensor]] = {
            "moving": {},
            "another": {},
            "transformed": {},
        }

        pad = lambda x: self.pad_fn(x.squeeze().unsqueeze(0)).float()

        moving_image_file, moving_seg_file = self.images[index], self.segs[index]
        moving_image_tio = tio.ScalarImage(self.path_to_images / moving_image_file)
        moving_seg_tio = tio.LabelMap(self.path_to_segmentations / moving_seg_file)

        randindex = random.randint(0, len(self.images) - 1)
        image_file, seg_file = self.images[randindex], self.segs[randindex]

        deformation_field = self.__get_deformation_field(moving_image_tio)
        deformation_field = self.pad_fn(deformation_field.squeeze()).unsqueeze(0).float()

        tsfm = lambda x: self.stn(
            x.squeeze().unsqueeze(0).unsqueeze(0).float(), deformation_field
        )

        # Expected format of the images is in B,C,D,W,H format
        # the pad function does all of the squeezing/unsqueezing necessary for 3D inputs
        moving_image = pad(moving_image_tio.data)
        moving_seg = pad(moving_seg_tio.data)
        transformed_image = pad(tsfm(moving_image))
        transformed_seg = pad(tsfm(moving_seg))
        another_image = pad(
            tio.ScalarImage(self.path_to_images / image_file).data
        )
        another_seg = pad(
            tio.LabelMap(self.path_to_segmentations / seg_file).data
        )

        data["moving"]["image"] = moving_image
        data["moving"]["seg"] = moving_seg

        data["transformed"]["image"] = transformed_image
        data["transformed"]["seg"] = transformed_seg
        data["transformed"]["field"] = deformation_field

        data["another"]["image"] = another_image
        data["another"]["seg"] = another_seg

        return data
