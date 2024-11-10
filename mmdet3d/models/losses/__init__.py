# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .paconv_regularization_loss import PAConvRegularizationLoss
from .kd_loss import (
    ChannelWiseDivergenceLoss,
    CriterionCWD_KD,
    SimpleMSE,
    SimpleBCE,
    MultiScaleFeatsDistill,
    MSEWithDist,
    MoCoC_Head_KD,
    InfoMax_Loss,
    VoxelDistill,
    FeatsNormDistill,
)


__all__ = [
    "FocalLoss",
    "SmoothL1Loss",
    "binary_cross_entropy",
    "ChamferDistance",
    "chamfer_distance",
    "axis_aligned_iou_loss",
    "AxisAlignedIoULoss",
    "PAConvRegularizationLoss",
    "ChannelWiseDivergenceLoss",
    "CriterionCWD_KD",
    "SimpleMSE",
    "SimpleBCE",
    "MultiScaleFeatsDistill",
    "MSEWithDist",
    "MoCoC_Head_KD",
    "InfoMax_Loss",
    "VoxelDistill",
    "FeatsNormDistill",
]
