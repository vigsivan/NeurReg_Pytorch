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
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.nn import Module

__all__ = ["SpatialTransformer3D", "RegistrationNetwork", "RegistrationSimulator3D"]

class SpatialTransformer3D(Module):
    """
    Note, this implementation follows the specification from the VoxelMorph Paper.
    """
    def __init__(self, use_cuda=False):
        self.device = "cuda" if use_cuda else "cpu"
        self.ndims = 3

    def forward(
        self, image: torch.Tensor, displacement_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Resamples input image onto the target grid.

        Recall:

        G_s -> Source grid
        G_t -> Target grid
        ẟ -> displacement field on G_s

        G_t = Φ(G_s) = G_s + ẟ(G_s)

        We then scale the targrt grid so that the inputs are within -1 and 1.

        G_ts = S(G_t)
        I_r = Resample(I, G_ts) where I and I_r are the original and resampled
        images respectively.

        Parameters
        ----------
        image: Torch.Tensor
            Image with dimensions (B, C, D, H, W)
        displacement_field: Torch.Tensor
            Displacement field with dimensions (B, D, H, W, 3)
        Returns
        -------
        transformed_image: Torch.Tensor
            Image with spatial transformer
        """
        _, _, D, H, W = image.shape
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.arange(0, D), torch.arange(0, H), torch.arange(0, W)
        )
        source_grid = torch.stack((grid_d, grid_h, grid_w), self.ndims).float() # (D, H, W, 3)
        source_grid.requires_grad = False

        target_grid = source_grid + displacement_field # (B, D, H, W, 3)
        tgrid = target_grid.view(-1,3)
        tgrid[:,0] = 2*(tgrid[:,0]-tgrid[:,0].min())/(tgrid[:,0].max()-tgrid[:,0].min())-1
        tgrid[:,1] = 2*(tgrid[:,1]-tgrid[:,1].min())/(tgrid[:,1].max()-tgrid[:,1].min())-1
        tgrid[:,2] = 2*(tgrid[:,2]-tgrid[:,2].min())/(tgrid[:,2].max()-tgrid[:,2].min())-1
        # d_min, d_max = tgrid_unrolled[:,0].min(), tgrid_unrolled[0].max()
        # h_min, h_max = tgrid_unrolled[:,1].min(), tgrid_unrolled[1].max()
        # w_min, w_max = tgrid_unrolled[:,2].min(), tgrid_unrolled[2].max()
        # # d_min, d_max = target_grid[:,:,:,0].min(), target_grid[:,:,:,0].max()
        # # h_min, h_max = target_grid[:,:,:,1].min(), target_grid[:,:,:,1].max()
        # # w_min, w_max = target_grid[:,:,:,2].min(), target_grid[:,:,:,2].max()
        # #
        # target_grid_scaled = torch.zeros_like(target_grid)
        # target_grid_scaled[:,:,:,0] = 2*((target_grid[:,:,:,0]-d_min)/(d_max-d_min))-1
        # target_grid_scaled[:,:,:,1] = 2*((target_grid[:,:,:,1]-h_min)/(h_max-h_min))-1
        # target_grid_scaled[:,:,:,2] = 2*((target_grid[:,:,:,2]-w_min)/(w_max-w_min))-1
        output = F.grid_sample(image, target_grid, align_corners=False) # (B, C, D, H, W)
        return output


class RegistrationNetwork(Module):
    def __init__(self):
        raise NotImplementedError


class RegistrationSimulator3D:
    """
    Parameters from the original paper are the default values.
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
        self.displacement_field = self.__get_displacement_field(image)
        return self.displacement_field

    def __get_displacement_field(self, image: sitk.Image) -> sitk.Image:

        # U: Uniformly sample
        U = lambda umin, umax: np.random.choice(np.linspace(umin, umax, 100))

        image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))

        image_fixed_parameters = []
        for param in [
            image.GetSize(),
            image.GetOrigin(),
            image.GetSpacing(),
            image.GetDirection(),
        ]:
            image_fixed_parameters.extend(param)

        #########################
        # 1. Rotation
        #########################

        angles = [U(0, self.rotation_max[i]) for i in range(self.ndims)]
        radians = [np.pi * angle / 180 for angle in angles]
        center = image.GetOrigin()
        rotation = sitk.Euler3DTransform(center, *radians, tuple([0] * self.ndims))
        rotation.SetFixedParameters(image_fixed_parameters)
        #########################
        # 2. Scaling
        #########################

        scaling = tuple(
            [U(smin, smax) for smin, smax in zip(self.scale_min, self.scale_max)]
        )

        scale = sitk.ScaleTransform(self.ndims, scaling)
        scale.SetFixedParameters(image_fixed_parameters)

        #########################
        # 3. Translation
        #########################

        translation_factor = [U(-i, i) for i in self.translation_factor]
        translation_amount = tuple(
            [image.GetSize()[i] * tf for i, tf in enumerate(translation_factor)]
        )
        translation = sitk.TranslationTransform(self.ndims, translation_amount)
        translation.SetFixedParameters(image_fixed_parameters)

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

        ctransform = sitk.CompositeTransform([translation, scale, rotation])
        ctransform = sitk.CompositeTransform(
            [elastic_displacement_field_transform, ctransform]
        )
        displacement_field = sitk.TransformToDisplacementField(
            ctransform,
            size=image.GetSize(),
            outputOrigin=image.GetOrigin(),
            outputSpacing=image.GetSpacing(),
            outputDirection=image.GetDirection(),
        )

        return displacement_field

def write_nifti(image: sitk.Image, fname: str):
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(fname+".nii.gz")
    writer.Execute(image)


if __name__ == "__main__":
    simulator = RegistrationSimulator3D()
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName("../data/test_image.nii.gz")
    image = reader.Execute()

    deformation_field = simulator.get_new_displacement_field(image)
    write_nifti(deformation_field, "deformation_field")

    deformation_array = sitk.GetArrayFromImage(deformation_field)
    deformation_pt = torch.Tensor(deformation_array).moveaxis(2,0).unsqueeze(0)
    image_pt = torch.Tensor(sitk.GetArrayFromImage(image)).moveaxis(-1,0).unsqueeze(0).unsqueeze(0)

    image_transformed_pt = SpatialTransformer3D().forward(image_pt, deformation_pt)
    image_transformed = sitk.GetImageFromArray(image_transformed_pt.squeeze().numpy().T)
    breakpoint()
    image_transformed.CopyInformation(image)
    write_nifti(image_transformed, "transformed")

    # writer.SetFileName("orig_array.nii.gz")
    # # writer.Execute(sitk.GetImageFromArray(image_pt.numpy()))
    #
    # image_transformed.SetSpacing(image.GetSpacing())
    # image_transformed.SetDirection(image.GetDirection())
    # image_transformed.SetOrigin(image.GetOrigin())
    #
    # writer.SetFileName("transformed.nii.gz")
    # writer.Execute(image_transformed)
    # writer.SetFileName("deformation_array.nii.gz")
    # writer.Execute(sitk.GetImageFromArray(deformation_array))
