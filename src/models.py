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

import os
from typing import Tuple, Union
import numpy as np
import SimpleITK as sitk
from torch.nn import Module

__all__ = ["SpatialTransformer", "RegistrationNetwork", "RegistrationSimulator3D"]


class SpatialTransformer(Module):
    def __init__(self):
        raise NotImplementedError


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


if __name__ == "__main__":
    simulator = RegistrationSimulator3D()
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName("../data/test_image.nii.gz")
    image = reader.Execute()

    deformation_field = simulator.get_new_displacement_field(image)
    # arr = sitk.GetArrayFromImage(deformation_field)
    # breakpoint()

    os.system("mkdir -p ../data/tmp")
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName("../data/tmp/deformation_field.nii.gz")
    writer.Execute(deformation_field)
    print(
        "Successfully generated deformation field at ../data/tmp/deformation_field.nii.gz"
    )
