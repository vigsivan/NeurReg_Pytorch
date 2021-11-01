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

"""
Problems:
* What happens when I programmatically resample the image using sitk?
* What happens when I programmatically resample the image using sitk after conversion to Pytorch?
"""

from typing import Tuple, Union
import random

import SimpleITK as sitk
import numpy as np
import torch
import torchio as tio
from torch.nn import Module

# __all__ = ["SpatialTransformer3D", "RegistrationNetwork", "RegistrationSimulator3D"]
#

# class SpatialTransformer3D(Module):
#     """
#     Note, this implementation follows the specification from the VoxelMorph Paper.
#
#     The SITK image is passed in so that we can convert index to physical coords
#     and resample the image in physical coords
#
#     Steps:
#         1. Compute image affine
#         2. Generate displaced grid
#         3. Resample image on this displaced grid
#
#     Potential problems:
#         index = Ainv * (P - Ori)
#     """
#
#     def __init__(self, use_cuda=False):
#         self.device = "cuda" if use_cuda else "cpu"
#         self.ndims = 3
#
#     def compute_physical_grid(self, image_sitk):
#         """
#         Computes the physical grid from the image
#         """
#         index_grid = torch.stack(
#             torch.meshgrid(*[torch.arange(0, i, 1) for i in image_sitk.GetSize()]),
#             dim=3,
#         ).float().reshape(3, -1)  # self.ndims).float()
#         orientation_matrix = np.array(image_sitk.GetDirection()).reshape(
#             (3, 3)
#         )  # self.ndims,self.ndims))
#         spacing_matrix = np.array(image_sitk.GetSpacing()) * np.identity(
#             3
#         )  # self.ndims)
#         image_affine = torch.tensor(orientation_matrix @ spacing_matrix).float()
#         physical_grid = torch.tensor(image_sitk.GetOrigin()) + (
#             image_affine @ index_grid
#         )
#         physical_grid.to(self.device)
#         return physical_grid
#
#     # def __normalize(self, grid) -> torch.Tensor:
#     #     """
#     #     torch's grid_simple takes inputs between -1 and 1, so this function
#     #     normalizes the inputs using the formula:
#     #
#     #     x = 2 * (x-minx)/(maxx-minx) -1
#     #     """
#     #     g = grid.view(3, -1)
#     #     for i in range(3):
#     #         x = g[i, :]
#     #         g[i, :] = 2 * (x - x.min()) / (x.max() - x.min()) - 1
#     #     return grid
#     #
#
#     def resample(self,image: torch.Tensor, grid: torch.Tensor)->torch.Tensor:
#         raise NotADirectoryError
#
#     def forward(
#         self,
#         image: torch.Tensor,
#         displacement_field: torch.Tensor,
#         image_sitk: sitk.Image,
#     ) -> torch.Tensor:
#         """
#         Resamples input image onto the target grid.
#
#         Recall:
#
#         G_s -> Source grid
#         G_t -> Target grid
#         ẟ -> displacement field on G_s
#
#         G_t = Φ(G_s) = G_s + ẟ(G_s)
#
#         We then scale the targrt grid so that the inputs are within -1 and 1.
#
#         G_ts = S(G_t)
#         I_r = Resample(I, G_ts) where I and I_r are the original and resampled
#         images respectively.
#
#         Parameters
#         ----------
#         image: Torch.Tensor
#             Image with dimensions (B, C, D, H, W)
#         displacement_field: Torch.Tensor
#             Displacement field with dimensions (B, D, H, W, 3)
#         sitk_image: sitk.Image
#             SimpleITK version of the image, which is used to generate a source grid
#         Returns
#         -------
#         transformed_image: Torch.Tensor
#             Image with spatial transformer
#         """
#         physical_grid = self.compute_physical_grid(image_sitk)
#         physical_grid.requires_grad = False
#
#         # ic(physical_grid.shape)
#         ic(displacement_field.shape)
#         target_grid = (
#             physical_grid + displacement_field # physical_grid + displacement_field
#         )  # (B, D, H, W, 3)
#         breakpoint()
#         ic(target_grid.shape)
#         # output = F.grid_sample(image, target_grid, mode="bilinear", align_corners=False)
#         output = self.resample(image, target_grid)
#
#         return output
#

class RegistrationNetwork(Module):
    def __init__(self):
        raise NotImplementedError


class RegistrationSimulator3D:
    """
    Parameters from the original paper are the default values.

    Note: the rotation parameter is in degrees.
    """

    def __init__(
        self,
        rotation_max: Union[float, Tuple[float, float, float]] = 30.0,
        scale_min: Union[float, Tuple[float, float, float]] = 0.75,
        scale_max: Union[float, Tuple[float, float, float]] = 1.25,
        translation_factor: Union[float, Tuple[float, float, float]] = 0.02,
        offset_gaussian_std_max: float = 1000,
        smoothing_gaussian_std_min: float = 10,
        smoothing_gaussian_std_max: float = 13,
    ):
        self.ndims = 3
        self.rotation_max = self.__tuplify(rotation_max)
        self.scale_min = self.__tuplify(scale_min)
        self.scale_max = self.__tuplify(scale_max)
        self.translation_factor = self.__tuplify(translation_factor)
        self.offset_gaussian_std_max = offset_gaussian_std_max
        self.smoothing_gaussian_std_min = smoothing_gaussian_std_min
        self.smoothing_gaussian_std_max = smoothing_gaussian_std_max

        self.transform = None
        self.deformation_field = None

    def __tuplify(
        self, a: Union[float, Tuple[float, float, float]]
    ) -> Tuple[float, float, float]:
        if isinstance(a, tuple) and len(a) != self.ndims:
            raise Exception(f"All tuple parameters should have length {self.ndims}")
        elif isinstance(a, tuple):
            return a
        else:
            return (a, a, a)

    def get_new_displacement_field(self, image: sitk.Image) -> sitk.Image:
        """
        Gets the new displacement field.
        """

        self.transform = self.__random_transform(image)
        t2df = sitk.TransformToDisplacementFieldFilter()
        t2df.SetReferenceImage(image)
        self.displacement_field = t2df.Execute(self.transform)
        return self.displacement_field

    def __random_transform(self, image: sitk.Image) ->  sitk.Transform:
        """
        Transforms the input grid image.

        This grid can then be used to resample the original image
        """

        # U: Uniformly sample
        U = lambda umin, umax: np.random.choice(np.linspace(umin, umax, 100))

        #########################
        # 1. Rotation
        #########################

        angles = [U(0, self.rotation_max[i]) for i in range(self.ndims)]
        angles = [U(0, 30) for _ in range(3)]
        radians = [np.pi * angle / 180 for angle in angles]
        center = image.GetOrigin()
        rotation = sitk.Euler3DTransform(center, *radians, tuple([0] * self.ndims))

        #########################
        # 2. Scaling
        #########################

        scaling = tuple(
            [U(smin, smax) for smin, smax in zip(self.scale_min, self.scale_max)]
        )

        scale = sitk.ScaleTransform(self.ndims, scaling)

        #########################
        # 3. Translation
        #########################

        translation_factor = [U(-i, i) for i in self.translation_factor]
        translation_amount = tuple(
            [image.GetSize()[i] * tf for i, tf in enumerate(translation_factor)]
        )
        translation = sitk.TranslationTransform(self.ndims, translation_amount)

        #########################
        # 4. Elastic Deformation
        #########################

        offset_std = U(0, self.offset_gaussian_std_max)
        smoothing_std = U(
            self.smoothing_gaussian_std_min, self.smoothing_gaussian_std_max
        )
        offset_field = np.zeros((*image.GetSize(), self.ndims))
        offset_field += np.random.normal(0, offset_std, offset_field.shape)

        # Pad the image to smooth the offset field
        # The minimum allowable pad is 2, we set it to 4 to avoid edge artifacts
        offset_field = np.pad(offset_field, pad_width=4)
        offset_field = sitk.GetArrayFromImage(
            sitk.SmoothingRecursiveGaussian(
                sitk.GetImageFromArray(offset_field), sigma=smoothing_std
            )
        )
        d = offset_field.shape
        offset_field = offset_field[4 : d[0] - 4, 4 : d[1] - 4, 4 : d[2] - 4, 4:7]

        offset_field = sitk.GetImageFromArray(offset_field, isVector=True)
        offset_field = sitk.Cast(offset_field, sitk.sitkVectorFloat64)
        offset_field.SetOrigin(image.GetOrigin())
        offset_field.SetDirection(image.GetDirection())
        offset_field.SetSpacing(image.GetSpacing())
        elastic_displacement_field_transform = sitk.DisplacementFieldTransform(
            offset_field
        )

        #########################
        # 5. Concatenate
        #########################

        # Note: SITK Composite Transform  applies transforms in reverse order,
        # so the transforms need to be added in reverse order

        ctransform = sitk.CompositeTransform(
            [elastic_displacement_field_transform, translation, scale, rotation]
        )
        return ctransform


def nifti_writer():
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")

    def write_nifti(image: Union[sitk.Image, torch.Tensor], fname: str):
        writer.SetFileName(fname + ".nii.gz")
        if isinstance(image, sitk.Image):
            writer.Execute(image)
        else:
            vector_image = sitk.GetImageFromArray(image.numpy())
            writer.Execute(vector_image)

    return write_nifti


## MAIN:
random.seed(42)
np.random.seed(42)

simulator = RegistrationSimulator3D()


reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName("../data/test_image.nii.gz")
image = reader.Execute()
writer = nifti_writer()

image_pt = torch.Tensor(sitk.GetArrayFromImage(image)).unsqueeze(0).unsqueeze(0)
im_reconstructed = sitk.GetImageFromArray(image_pt.squeeze().numpy())

displacement_field = simulator.get_new_displacement_field(image)
displacement_field_pt = (
    torch.Tensor(sitk.GetArrayFromImage(displacement_field)).reshape((*image_pt, 3)).unsqueeze(0)
)


writer(displacement_field_pt, "displacement_field_pt")
writer(image_pt, "image_pt")

index_grid = torch.stack(
    torch.meshgrid(*[torch.arange(0, i, 1) for i in image_pt.squeeze().shape]),
    dim=3,
).float().reshape(3, -1)  # self.ndims).float()

orientation_matrix = np.array(image.GetDirection()).reshape(
    (3, 3)
)  # self.ndims,self.ndims))
spacing_matrix = np.array(image.GetSpacing()) * np.identity(
    3
)  # self.ndims)
image_affine = torch.tensor(orientation_matrix @ spacing_matrix).float()
physical_grid = image_affine @ index_grid
physical_grid = physical_grid.reshape_as(displacement_field_pt)
physical_grid += torch.tensor(image.GetOrigin())

target_grid = physical_grid + displacement_field_pt

# writer(displacement_field, "displacement_field")
#
# image_transformed = sitk.GetImageFromArray(image_transformed_pt.squeeze().numpy().T)
# nriter(image_transformed, "transformed")

# Junk code:
# Make quiver plot of displacement_field
# slice_index = dfarr.shape[0] // 2
# df_slice = dfarr[slice_index, :, :, :]
#
# orientation = df.GetDirection()
# spacing = df.GetSpacing()
#
# size = image_sitk.GetSize()
#
# orientation = np.array(orientation).reshape((3, 3))
# voxel_to_physical = (spacing * np.identity(3)) @ orientation
# physical_to_voxel = np.linalg.inv(voxel_to_physical)
#
# # for i in range(df_slice.shape[0]):
# #     for j in range(df_slice.shape[1]):
# #         p = physical_to_voxel @ df_slice[i,j]
# #         df_slice[i,j] = p
#
# x = df_slice[:, :, 0][::-1, ::-1]
# y = -df_slice[:, :, 1][::-1, ::-1]
#
# coordsX = np.arange(
#     0, size[0] * spacing[0], size[0] * spacing[0] / float(image.shape[2])
# )
# coordsY = np.arange(
#     0, size[1] * spacing[1], size[1] * spacing[1] / float(image.shape[1])
# )
#
# coordsX, coordsY = np.meshgrid(coordsX, coordsY)
#
# M = np.sqrt(x * x + y * y)
#
# qq = plt.quiver(coordsX, coordsY, x, y, M, cmap=plt.cm.jet, units="x", scale=1)
#
# plt.axis("off")
# plt.show()
