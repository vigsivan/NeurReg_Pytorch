"""
NeurReg Models
--------------

This file defines the following "modules":

* RegistrationSimulator3D (S)  <- not a PyTorch module
* SpatialTransformer (τ)       <- Pytorch modules
* RegistrationNetwork (N)      <-       ︙

The inputs and outputs are denoted and defined as follows:

* M: atlas
    We use this as the moving image (the image that gets deformably registered)
* F_g: registration_ground_truth
    F_g = S(φ|M), where φ are the deformation params we set
* I: transformed_atlas
    I = τ(M, F_g)
* F: learned_registration
    F = N(M, I; θ), where θ are parameters of the network N.
"""

from typing import Tuple, Union
import random

import SimpleITK as sitk
import numpy as np
import torch
import torchio as tio
from torch.nn import Module
import torch.nn.functional as F

__all__ = ["SpatialTransformer", "RegistrationNetwork", "RegistrationSimulator3D"]


class RegistrationNetwork(Module):
    def __init__(self):
        raise NotImplementedError


class SpatialTransformer(Module):
    """
    N-D Spatial Transformer

    Pulled from VoxelMorph Pytorch implementation
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class RegistrationSimulator3D:
    """
    Parameters from the original paper are the default values.
    The elastic deformation is using bspline registration.

    Note: the rotation parameter is in degrees.
    """

    def __init__(
        self,
        rotation_min: Union[float, Tuple[float, float, float]] = 0.0,
        rotation_max: Union[float, Tuple[float, float, float]] = 30.0,
        scale_min: Union[float, Tuple[float, float, float]] = 0.75,
        scale_max: Union[float, Tuple[float, float, float]] = 1.25,
        translation_factor: Union[float, Tuple[float, float, float]] = 0.02,
        control_points: Union[int, Tuple[int, int, int]] = 7
        # offset_gaussian_std_max: float = 1000,
        # smoothing_gaussian_std_min: float = 10,
        # smoothing_gaussian_std_max: float = 13,
    ):
        self.ndims = 3
        self.rotation_min = self.__tuplify(rotation_min)
        self.rotation_max = self.__tuplify(rotation_max)
        self.scale_min = self.__tuplify(scale_min)
        self.scale_max = self.__tuplify(scale_max)
        self.translation_factor = self.__tuplify(translation_factor)
        self.control_points = self.__tuplify_int(control_points)
        # self.offset_gaussian_std_max = offset_gaussian_std_max
        # self.smoothing_gaussian_std_min = smoothing_gaussian_std_min
        # self.smoothing_gaussian_std_max = smoothing_gaussian_std_max
        self.displacement_field: tio.ScalarImage

    def __tuplify(
        self, x: Union[float, Tuple[float, ...]]
    ) -> Tuple[float, float, float]:
        """
        A necessary evil to get around indeterminate Tuple length errors in Pyright.

        Solves the problem of:
            assert len(some_list) == 3
            x: Tuple[t,t,t] = tuple(some_list) <- Pyright throws an error here
        """
        if isinstance(x, tuple):
            assert len(x) == 3
            return (x[0], x[1], x[2])
        return (x, x, x)

    # FIXME: look into generics
    def __tuplify_int(self, x: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int]:
        """
        A necessary evil to get around indeterminate Tuple length errors in Pyright.

        Solves the problem of:
            assert len(some_list) == 3
            x: Tuple[t,t,t] = tuple(some_list) <- Pyright throws an error here
        """
        if isinstance(x, tuple):
            assert len(x) == 3
            return (x[0], x[1], x[2])
        return (x, x, x)

    def get_new_displacement_field(self, image: tio.ScalarImage) -> tio.Image:
        """
        Gets the new displacement field.
        """

        self.transform = self.__random_transform(image)
        t2df = sitk.TransformToDisplacementFieldFilter()
        t2df.SetReferenceImage(image.as_sitk())
        self.displacement_field = tio.ScalarImage.from_sitk(
            t2df.Execute(self.transform)
        )
        return self.displacement_field

    def __random_transform(self, image: tio.ScalarImage) -> sitk.Transform:
        """
        Transforms the input grid image.

        This grid can then be used to resample the original image
        """

        U = lambda umin, umax: np.random.choice(np.linspace(umin, umax, 100))

        scales = self.__tuplify(
            tuple([U(smin, smax) for smin, smax in zip(self.scale_min, self.scale_max)])
        )
        rotations = self.__tuplify(
            tuple(
                [
                    U(rmin, rmax)
                    for rmin, rmax in zip(self.rotation_min, self.rotation_max)
                ]
            )
        )
        tfs = [U(-i, i) for i in self.translation_factor]
        translation = self.__tuplify(
            tuple([image.spatial_shape[i] * tf for i, tf in enumerate(tfs)])
        )

        affine_tio = tio.Affine(
            scales=scales,
            degrees=rotations,
            translation=translation,
        )
        affine_sitk = affine_tio.get_affine_transform(image)

        elastic_cps = tio.RandomElasticDeformation.get_params(
            num_control_points=self.control_points,
            max_displacement=translation,
            num_locked_borders=2,
        )
        elastic = tio.ElasticDeformation(
            control_points=elastic_cps, max_displacement=translation
        )
        elastic_sitk = elastic.get_bspline_transform(image.as_sitk())

        return sitk.CompositeTransform((elastic_sitk, affine_sitk))


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    simulator = RegistrationSimulator3D()
    image = tio.ScalarImage("../data/test_image.nii.gz")
    disp_field = simulator.get_new_displacement_field(image)

    image_tensor = image.data.unsqueeze(0)
    disp_tensor = disp_field.data.unsqueeze(0)

    stn = SpatialTransformer(image_tensor.squeeze().shape)
    resampled = stn.forward(image_tensor, disp_tensor).squeeze()
    resampled_image = tio.ScalarImage(tensor=resampled.unsqueeze(0))
    resampled_image.save("../data/resamped.nii.gz")
