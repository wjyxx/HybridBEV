# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
import torch
import numpy as np
from torch import nn as nn
from functools import partial

from mmdet3d.ops import make_sparse_convmodule

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from .. import builder
from ..builder import MIDDLE_ENCODERS
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.models.fusion_layers.dynamic_fusion.focal_sparse_conv import (
    FocalSparseConv,
)


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, gt_boxes=None):
        loss_box_of_pts = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, _loss = module(input, gt_boxes)
                loss_box_of_pts += _loss
            else:
                input = module(input)
        return input, loss_box_of_pts


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


@MIDDLE_ENCODERS.register_module()
class SparseEncoderHD(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        **kwargs,
    ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False

        # dynamic fusion voxelprojected
        self.dynamic_config = kwargs["dynamic_config"]
        try:
            self.dynamic_fusion = kwargs["dynamic_fusion"]
            self.image_list = self.dynamic_fusion["image_list"]
        except Exception as e:
            self.dynamic_fusion = False

        # Spconv init all weight on its own
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.dynamic_fusion:
            self.conv_focal_multimodal = FocalSparseConv(
                16,
                16,
                voxel_stride=1,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="spconv_focal_multimodal",
                skip_loss=self.dynamic_config["skip_loss"],
                mask_multi=self.dynamic_config["mask_multi"],
                topk=self.dynamic_config["topk"],
                threshold=self.dynamic_config["threshold"],
                use_img=self.dynamic_config["use_img"],
            )
            self._dynamic_fusion = builder.build_fusion_layer(self.dynamic_fusion)
            # self.fusion = builder.build_fusion_layer(dynamic_fusion)

        special_spconv_fn = partial(
            FocalSparseConv,
            skip_loss=self.dynamic_config["skip_loss"],
            enlarge_voxel_channels=self.dynamic_config["enlarge_voxel_channels"],
            mask_multi=self.dynamic_config["mask_multi"],
            topk=self.dynamic_config["topk"],
            threshold=self.dynamic_config["threshold"],
        )

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True),
        )

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            special_spconv_fn(
                16,
                16,
                kernel_size=3,
                voxel_stride=1,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="focal0",
            )
            # if 1 in special_conv_list else None,
        )

        self.conv2 = SparseSequentialBatchdict(
            spconv.SparseSequential(
                SparseConv3d(
                    16, 32, 3, 2, padding=1, bias=False
                ),  # [1600, 1200, 41] -> [800, 600, 21]   1/2
                build_norm_layer(norm_cfg, 32)[1],
                nn.ReLU(inplace=True),
            ),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = SparseSequentialBatchdict(
            spconv.SparseSequential(
                SparseConv3d(
                    32, 64, 3, 2, padding=1, bias=False
                ),  # [800, 600, 21] -> [400, 300, 11]  1/4
                build_norm_layer(norm_cfg, 64)[1],
                nn.ReLU(inplace=True),
            ),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = SparseSequentialBatchdict(
            spconv.SparseSequential(
                SparseConv3d(
                    64, 128, 3, 2, padding=[0, 1, 1], bias=False
                ),  # [400, 300, 11] -> [200, 150, 5] 1/8
                build_norm_layer(norm_cfg, 128)[1],
                nn.ReLU(inplace=True),
            ),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]   1/16
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

        # self.distill_conv_out = spconv.SparseSequential(
        #     SparseConv3d(
        #         16, 16, 3, bias=False
        #     ),  # [41,1024,1024] - [41,1024,1024]   1/1
        #     build_norm_layer(norm_cfg, 16)[1],
        #     nn.ReLU(),
        # )

    def batch_data_fit(self, batch_data):
        """
        Args:
            batch_data(dict[gt_boxes, img_metas, cam_intrinsic, img_feat])
        """

        keys = batch_data.keys()
        assert "gt_boxes" in keys, "loss key of gt_boxes"
        assert "img_metas" in keys, "loss key of img_metas"
        assert "cam_intrinsic" in keys, "loss key of cam_intrinsic"
        # assert 'img_feat' in keys, 'loss key of img_feat'

        # image_list = [x.lower() for x in self.image_list]

        _img_metas = batch_data["img_metas"]
        img_shape = []
        lidar2img = []
        lidar2cam = []
        resizes = []
        raw_img_shapes = []

        # img_metas = batch_data['img_metas'][0]
        # from termcolor import colored
        # print(colored(_img_metas[0].keys(), 'green'))
        # print(colored(img_metas['pcd_horizontal_flip'], 'red'))
        # print(colored(img_metas['pcd_vertical_flip'], 'yellow'))
        # print(colored(f"{img_metas['lidar2cam'],type(img_metas['lidar2cam'])}", 'blue'))

        B = len(batch_data["img_metas"])
        for idx, x in enumerate(_img_metas):
            img_shape.append(
                torch.stack(x["img_shape"]).unsqueeze(0)
            )  # list to tensor(1 n 2)
            lidar2img.append(torch.from_numpy(np.array(x["lidar2img"])).unsqueeze(0))
            lidar2cam.append(torch.from_numpy(np.array(x["lidar2cam"])).unsqueeze(0))
            resizes.append(torch.from_numpy(np.array(x["resize"])).unsqueeze(0))
            raw_img_shapes.append(torch.stack(x["raw_img_shape"]).unsqueeze(0))

        img_shape = torch.cat(img_shape)  # to b n 2
        lidar2img = torch.cat(lidar2img)  # to b n 4 4
        lidar2cam = torch.cat(lidar2cam)  # to b n 4 4
        resize = torch.cat(resizes)  # b n 1
        raw_img_shape = torch.cat(raw_img_shapes)  # b n 2

        # print(colored(f"{img_shape.shape,lidar2img.shape,lidar2cam.shape}", 'yellow'))
        cam_intrinsic = batch_data["cam_intrinsic"]  # b n 3 3
        img_feats = batch_data["img_feat"] if "img_feat" in keys else None

        new_batch_data = {
            "img_shape": img_shape,
            "lidar2img": lidar2img,
            "lidar2cam": lidar2cam,
            "cam_intrinsic": cam_intrinsic,
            "img_feat": img_feats,
            "gt_boxes": batch_data["gt_boxes"],
            "resize": resize,
            "raw_img_shape": raw_img_shape,
        }

        return new_batch_data

    @auto_fp16(apply_to=("voxel_features",))
    def forward(
        self, voxel_features, coors, batch_size, img_feats, batch_data_middle_encoder
    ):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            batch_data_middle_encoder(dict[gt_boxes, img_shape, lidar2img, lidar2cam, cam_intrinsic]): Batch img metas params

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )

        multi_scale_voxel_features = {}
        # print(f"==={input_sp_tensor.features.shape}====")

        # NOTE C (<16>, 32, 64, 128, 128)

        x = self.conv_input(input_sp_tensor)  # shape of b c d h w: [41, 1024, 1024]

        loss_box_of_pts = 0

        # update img_feats params
        batch_data_middle_encoder["img_feat"] = img_feats
        # batch_data_middle_encoder = self.batch_data_fit(batch_data_middle_encoder)

        x_conv1, _loss = self.conv1(
            x, batch_data_middle_encoder
        )  # input size equal to the input size, and nothing changed
        # return x_conv1_ret

        loss_box_of_pts += _loss

        if self.dynamic_fusion:
            x_conv1, _loss = self.conv_focal_multimodal(
                x_conv1, batch_data_middle_encoder, fuse_func=self._dynamic_fusion
            )  # [1440, 1440, 40]

        # Todo: adjust the batch_data to fit the format of the following dynamic fusion module

        x_conv2, _loss = self.conv2(
            x_conv1, batch_data_middle_encoder
        )  # [1440, 1440, 40]  -> [720, 720, 20]
        loss_box_of_pts += _loss

        x_conv3, _loss = self.conv3(
            x_conv2, batch_data_middle_encoder
        )  # [720, 720, 20] -> [360, 360, 10]
        loss_box_of_pts += _loss

        x_conv4, _loss = self.conv4(
            x_conv3, batch_data_middle_encoder
        )  # [360, 360, 10] -> [180, 180, 5]
        loss_box_of_pts += _loss

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        # x_conv1 = self.distill_conv_out(x_conv1)
        # x_conv1 = x_conv1.dense()

        multi_scale_voxel_features = {
            "conv1": x_conv1,
            # "conv2": x_conv2,
            # "conv3": x_conv3,
            # "conv4": x_conv4,
        }

        return ret, multi_scale_voxel_features, loss_box_of_pts

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequentialBatchdict(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
