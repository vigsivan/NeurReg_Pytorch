"""
NeurReg Components
------------------

This file defines the following components:

* RegistrationSimulator3D (S)
* SpatialTransformer (τ)
* RegistrationNetwork (N)

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

from typing import Tuple, Union, List
import random

import SimpleITK as sitk
import elasticdeform
import numpy as np
import torch
import torchio as tio
from torch import nn
import torch.nn.functional as F

__all__ = ["SpatialTransformer", "Unet3D", "RegistrationSimulator3D"]

default_encoder_decoder_features = [[16, 32, 32, 32], [32, 32, 32, 32, 16, 16]]


class Unet3D(nn.Module):
    """
    Adapted from VoxelMorph repo at https://github.com/voxelmorph/voxelmorph.

    The default parameters are those used in the paper.

    Parameters
    ----------
    inshape: Tuple[int,int,int]
        Spatial dimensions of the input image.
    infeats: int
         Number of input features (channel dimension).
         Default: 2
    nb_features: List[List[int]]
        Unet convolutional features. Specified via a list of lists with
        the form [[encoder feats], [decoder feats]]
        Default: [[16, 32, 32, 32], [32, 32, 32, 32, 16, 16]]
    nb_conv_per_level: int
        Number of convolutions per unet level.
        Default: 1
    half_res: bool
        Skip the last decoder upsampling.
        Default: False
    """

    def __init__(
        self,
        inshape: Tuple[int, int, int],
        infeats: int = 2,
        nb_features: List[List[int]] = default_encoder_decoder_features,
        max_pool=2,
        nb_conv_per_level=1,
        half_res: bool = False,
    ):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for _, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer

    Pulled from VoxelMorph Pytorch implementation

    We also use this module to generate the elastic deformation field
    as we would like this to happen on the GPU.
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.float()


        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow) -> torch.Tensor:
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


class RegistrationSimulator3D(nn.Module):
    """
    Parameters from the original paper are the default values.
    The elastic deformation is using bspline registration.

    Note: the rotation parameter is in degrees.
    """

    def __init__(
        self,
        use_cuda: bool = False,
        rotation_min: Union[float, Tuple[float, float, float]] = 0.0,
        rotation_max: Union[float, Tuple[float, float, float]] = 30.0,
        scale_min: Union[float, Tuple[float, float, float]] = 0.75,
        scale_max: Union[float, Tuple[float, float, float]] = 1.25,
        translation_factor: Union[float, Tuple[float, float, float]] = 0.02,
        offset_gaussian_std_max: int = 1000,
        smoothing_gaussian_std_min: float = 10,
        smoothing_gaussian_std_max: float = 13,
    ):

        super().__init__()
        self.ndims = 3
        self.rotation_min = self.__tuplify(rotation_min)
        self.rotation_max = self.__tuplify(rotation_max)
        self.scale_min = self.__tuplify(scale_min)
        self.scale_max = self.__tuplify(scale_max)
        self.translation_factor = self.__tuplify(translation_factor)

        self.offset_gaussian_std_max = offset_gaussian_std_max
        self.smoothing_gaussian_std_min = smoothing_gaussian_std_min
        self.smoothing_gaussian_std_max = smoothing_gaussian_std_max
        self.use_cuda = use_cuda

        self.offset_gaussian_std_max = offset_gaussian_std_max

        U = lambda umin, umax: np.random.choice(np.linspace(umin, umax, 100)).item()
        smoothing_std = U(smoothing_gaussian_std_min, smoothing_gaussian_std_max)
        self.conv = nn.Conv3d(3, 3, int(smoothing_std), padding=1)

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

    def get_new_displacement_field(self, image: tio.ScalarImage) -> torch.Tensor:
        """
        Gets the new displacement field.
        """

        U = lambda umin, umax: np.random.choice(np.linspace(umin, umax, 100)).item()

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

        t2df = sitk.TransformToDisplacementFieldFilter()
        t2df.SetReferenceImage(image.as_sitk())
        disp_field = tio.ScalarImage.from_sitk(
            t2df.Execute(
                tio.Affine(
                    scales=scales, degrees=rotations, translation=translation
                ).get_affine_transform(image)
            )
        ).data

        elastic_offset_std = U(0, self.offset_gaussian_std_max)
        coarse_elastic_disp = torch.empty_like(disp_field).normal_(
            0, elastic_offset_std
        )

        if self.use_cuda:
            disp_field = disp_field.cuda()
            coarse_elastic_disp = coarse_elastic_disp.cuda()

            disp_field.requires_grad = False
            coarse_elastic_disp.requires_grad = False

        smooth_elastic = self.conv(coarse_elastic_disp.unsqueeze(0))
        return disp_field + smooth_elastic

    def forward(self, image: tio.ScalarImage) -> torch.Tensor:
        return self.get_new_displacement_field(image)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    image = tio.ScalarImage("../data/test_image.nii.gz")
    simulator = RegistrationSimulator3D()
    disp_tensor = simulator.get_new_displacement_field(image).unsqueeze(0).float()

    # image_tensor = image.data.unsqueeze(0).float()
    # #
    # stn = SpatialTransformer(image_tensor.squeeze().shape)
    # resampled = stn.forward(image_tensor, disp_tensor).squeeze()
    # resampled_image = tio.ScalarImage(tensor=resampled.unsqueeze(0))
    # resampled_image.save("../data/resampled3.nii.gz")
