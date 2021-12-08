from components import *
from dataclasses import dataclass
from typing import (
    Optional,
    Tuple,
    Union,
)

import torch
from torch.nn import (
    Module,
    Sequential,
    Conv3d,
    Softmax,
    Parameter,
)
from torch.distributions import Normal

__all__ = ["NeurRegNet"]


@dataclass
class NeurRegTrainOutputs:
    """
    Data class for storing the outputs of the model.
    """

    moving_to_target_field: torch.Tensor
    moving_to_target_image: torch.Tensor
    moving_to_target_segmentation: torch.Tensor
    moving_to_precomputed_field: torch.Tensor
    moving_to_precomputed_image: torch.Tensor
    moving_to_precomputed_segmentation: torch.Tensor
    precomputed_image: torch.Tensor
    precomputed_segmentation: torch.Tensor


@dataclass
class NeurRegValOutputs:
    """
    Data class for storing the outputs of the model during inference.
    """

    moving_to_target_field: torch.Tensor
    moving_to_target_image: torch.Tensor
    moving_to_target_segmentation: torch.Tensor


class NeurRegNet(Module):
    """
    NeurRegNet

    Parameters
    ----------
    target_shape: Tuple[int,int,int]
        The shape of the inputs
    """

    def __init__(
        self, target_shape: Tuple[int, int, int]
    ):
        super().__init__()
        self.N = Unet3D(inshape=target_shape)
        self.stn = SpatialTransformer(target_shape)

        # TODO: review the to_flow_field code
        self.conv_w_softmax = Sequential(Conv3d(17, 1, 3, padding=1), Softmax(3))
        self.to_flow_field = Conv3d(16, 3, 3, padding=1, bias=True)
        self.to_flow_field.weight = Parameter(
            Normal(0, 1e-5).sample(self.to_flow_field.weight.shape)
        )

        if self.to_flow_field.bias is not None:
            self.to_flow_field.bias = Parameter(
                torch.zeros(self.to_flow_field.bias.shape)
            )

    def forward(
        self,
        moving_image: torch.Tensor,
        moving_seg: torch.Tensor,
        target_image: torch.Tensor,
        transformed_image: Optional[torch.Tensor] = None,
        transformed_seg: Optional[torch.Tensor] = None,
    ) -> Union[NeurRegTrainOutputs, NeurRegValOutputs]:
        """
        Parameters
        ----------
        moving_image: torch.Tensor,
        moving_seg: torch.Tensor,
        target_image: torch.Tensor,
        target_seg: Optional[torch.Tensor],
        transform_field: Optional[torch.Tensor],
            Default=False

        Returns
        -------
        outputs: NeurRegOutputs
        """

        image_concat = torch.cat((moving_image, target_image), 1)
        moving_to_target_field = self.to_flow_field(self.N(image_concat))
        moving_to_target_image = self.stn(moving_image, moving_to_target_field)
        moving_to_target_seg = self.stn(moving_seg, moving_to_target_field)

        if self.training:

            assert transformed_image is not None and transformed_seg is not None
            transform_concat = torch.cat((moving_image, transformed_image), 1)

            last_layer = self.N(transform_concat)  # cache last layer for boosting
            moving_to_precomputed_field = self.to_flow_field(last_layer)
            moving_to_precomputed_image = self.stn(
                moving_image, moving_to_precomputed_field
            )
            moving_to_precomputed_segmentation = self.stn(
                moving_seg, moving_to_precomputed_field
            )
            # Concatenate along the channel dimension
            boosted = torch.cat((last_layer, moving_to_precomputed_segmentation), dim=1)
            boosted_segmentation = self.conv_w_softmax(boosted)

            outputs = NeurRegTrainOutputs(
                moving_to_target_field=moving_to_target_field,
                moving_to_target_image=moving_to_target_image,
                moving_to_target_segmentation=moving_to_target_seg,
                moving_to_precomputed_field=moving_to_precomputed_field,
                moving_to_precomputed_image=moving_to_precomputed_image,
                moving_to_precomputed_segmentation=boosted_segmentation,
                precomputed_image=transformed_image,
                precomputed_segmentation=transformed_seg,
            )

            return outputs

        else:
            return NeurRegValOutputs(
                moving_to_target_field=moving_to_target_field,
                moving_to_target_image=moving_to_target_image,
                moving_to_target_segmentation=moving_to_target_seg,
            )
