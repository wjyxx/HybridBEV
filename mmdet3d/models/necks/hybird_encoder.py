from curses import raw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init, build_norm_layer
from mmcv.runner.base_module import BaseModule
from ..builder import NECKS
from mmcv.cnn.bricks import ConvModule
from matplotlib import pyplot as plt

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except:
    import spconv
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d


class SCConv(nn.Module):
    def __init__(self, input_channels, output_channels, pooling_r, norm_layer=nn.ReLU):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, stride=1, padding=1
            ),
            norm_layer(output_channels),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, stride=1, padding=1
            ),
            norm_layer(output_channels),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, stride=1, padding=1
            ),
            norm_layer(output_channels),
        )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])) + 1e-8
        )  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    """SCNet SCBottleneck"""

    def __init__(
        self, input_channels, output_channels, pooling_r=4, norm_layer=nn.BatchNorm2d
    ):
        super(SCBottleneck, self).__init__()
        group_width = int(input_channels / 2)
        self.conv1_a = nn.Conv2d(input_channels, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(input_channels, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)

        self.k1 = nn.Sequential(
            nn.Conv2d(group_width, group_width, kernel_size=3, padding=1, stride=1),
            norm_layer(group_width),
        )

        self.scconv = SCConv(
            group_width, group_width, pooling_r=pooling_r, norm_layer=norm_layer
        )

        self.conv3 = nn.Conv2d(
            group_width * 2, output_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


@NECKS.register_module()
class SCNET(nn.Module):
    def __init__(self, input_channels, num_layers=5):
        super().__init__()

        self.layers = self.make_layer(
            SCBottleneck, input_channels, num_blocks=num_layers
        )

    def forward(self, bev_feats):
        """
        Args:
            bev_feats
        Returns:
        """
        x = bev_feats.float()
        x = self.layers(x)

        return x

    def make_layer(self, block, input_channels, num_blocks, norm_layer=nn.BatchNorm2d):

        layers = []

        for i in range(1, num_blocks):
            layers.append(block(input_channels, input_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)


@NECKS.register_module()
class HybirdEncoder(BaseModule):
    """Implements the view transformer."""

    def __init__(
        self,
        num_cams=6,
        num_convs=3,
        num_points=5,
        num_feature_levels=4,
        num_sweeps=1,
        cam_sweep_feq=12,
        kernel_size=(1, 3, 3),
        sweep_fusion=dict(type="sweep_cat"),
        keep_sweep_dim=False,
        embed_dims=128,
        norm_cfg=None,
        use_for_distill=False,
        encoding=True,
        conv_dict=None,
        **kwargs,
    ):
        super(HybirdEncoder, self).__init__()
        self.conv_layer = []
        fp16_enabled = kwargs.get("fp16_enabled", False)
        if norm_cfg is None:
            norm_cfg = kwargs.get("norm_cfg", dict(type="BN"))
        self.num_sweeps = num_sweeps
        self.sweep_fusion = sweep_fusion.get("type", "")
        self.keep_sweep_dim = keep_sweep_dim
        self.use_for_distill = use_for_distill
        self.num_cams = num_cams
        self.depth_proj = ReuseDepth(
            embed_dims=embed_dims,
            num_levels=num_feature_levels,
            num_points=num_points,
            num_cams=num_cams,
            num_sweeps=num_sweeps,
            **kwargs,
        )
        self.encoding = encoding

        if "sweep_cat" in self.sweep_fusion and self.encoding:
            self.trans_conv = nn.Sequential(
                nn.Conv3d(
                    embed_dims * 4,
                    embed_dims,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.BatchNorm3d(embed_dims),
                nn.ReLU(inplace=True),
            )

        self.num_feature_levels = num_feature_levels
        padding = tuple([(_k - 1) // 2 for _k in kernel_size])

        # for k in range(num_convs):
        #     conv = nn.Sequential(
        #         nn.Conv3d(
        #             embed_dims,
        #             embed_dims,
        #             kernel_size=kernel_size,
        #             stride=1,
        #             padding=padding,
        #             bias=True,
        #         ),
        #         nn.BatchNorm3d(embed_dims),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.conv_layer.append(conv)

        if fp16_enabled:
            self.fp16_enabled = True

    # @auto_fp16(apply_to=("mlvl_feats"))
    def forward(self, mlvl_feats, **kwargs):
        """Forward function for `Uni3DViewTrans`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
        """
        # Sweep number and Cam number could be dynamic
        if self.num_sweeps > 1:
            num_sweep, num_cam = kwargs["img_metas"][0]["sweeps_ids"].shape
        else:
            num_sweep = self.num_sweeps
            num_cam = self.num_cams

        kwargs["num_sweep"] = num_sweep
        kwargs["num_cam"] = num_cam
        # kwargs["batch_size"] = len(mlvl_feats[0])

        if self.num_feature_levels == 1:
            mlvl_feats = mlvl_feats[None, ...]

        # with torch.no_grad():
        voxel_space = self.depth_proj(
            mlvl_feats, img_depth=kwargs.pop("img_depth"), **kwargs
        )

        if self.encoding:
            voxel_space = self.feat_encoding(voxel_space, **kwargs)

        return voxel_space

    def feat_encoding(self, voxel_space, **kwargs):
        num_sweep = kwargs["num_sweep"]

        if "sweep_sum" in self.sweep_fusion:
            voxel_space = voxel_space.reshape(-1, num_sweep, *voxel_space.shape[1:])
            voxel_space = voxel_space.sum(1)
            num_sweep = 1

        elif "sweep_cat" in self.sweep_fusion:
            voxel_space = voxel_space.reshape(
                -1, num_sweep * voxel_space.shape[1], *voxel_space.shape[2:]
            )
            voxel_space = self.trans_conv(voxel_space)
            num_sweep = 1

        return voxel_space


class ReuseDepth(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        num_levels=4,
        num_points=5,
        num_cams=6,
        num_sweeps=1,
        pc_range=None,
        voxel_shape=None,
        device="cuda",
        fp16_enabled=False,
    ):
        super(ReuseDepth, self).__init__()
        self.device = device
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.num_sweeps = num_sweeps
        # build voxel space with X,Y,Z
        _width = torch.linspace(0, 1, self.voxel_shape[0], device=self.device)
        _hight = torch.linspace(0, 1, self.voxel_shape[1], device=self.device)
        _depth = torch.linspace(0, 1, self.voxel_shape[2], device=self.device)
        self.reference_voxel = torch.stack(
            torch.meshgrid([_width, _hight, _depth]), dim=-1
        )
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def forward(self, mlvl_feats, img_depth=None, **kwargs):
        bs = kwargs.get("batch_size", 1)
        num_sweep = kwargs.get("num_sweep", 1)
        num_cam = kwargs.get("num_cam", 6)
        reference_voxel = self.reference_voxel.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        fp16_enabled = True if hasattr(self, "fp16_enabled") else False

        output, depth, mask, reference_voxel = feature_sampling(
            mlvl_feats,
            reference_voxel,
            self.pc_range,
            kwargs["img_metas"],
            img_depth,
            num_sweep,
            num_cam,
            fp16_enabled,
            kwargs["img_shape"],
            kwargs["post_rots"],
            kwargs["post_trans"],
        )

        output = output.reshape(
            *output.shape[:2], -1, self.num_points, *output.shape[-3:]
        )  # b 1 xy z n 1 4
        mask = mask.reshape(
            *mask.shape[:2], -1, self.num_points, *mask.shape[-3:]
        )  # b 1 xy z n 1 1
        depth = depth.reshape(
            *depth.shape[:2], -1, self.num_points, *depth.shape[-3:]
        )  # b 1 xy z n 1 4
        depth = depth * mask  # b 1 xy z n 1 4
        output = output * depth  # b 1 xy z n 1 4

        if self.num_sweeps == 1:
            output = output.view(*output.shape[:4], -1).sum(
                -1
            )  # b 1 xy z n4 -> b 1 xy z 1
            # shape: (N, C, W, H, D)
            output = output.reshape(*output.shape[:2], *self.voxel_shape)  # b 1 x y z
            # reshape to (N, C, D, H, W)
            output = output.permute(0, 1, 4, 3, 2)  # b 1 z y x
        else:
            output = output.reshape(*output.shape[:4], num_cam, num_sweep, -1)
            output = (
                output.transpose(-2, -1)
                .reshape(*output.shape[:4], -1, num_sweep)
                .sum(-2)
            )
            # shape: (N, C, W, H, D, S)
            output = output.reshape(*output.shape[:2], *self.voxel_shape, num_sweep)
            # permute to (N, S, C, D, H, W)
            output = output.permute(0, 5, 1, 4, 3, 2)
            # reshape to (N*S, C, D, H, W)
            output = output.reshape(-1, *output.shape[2:])

        return output.contiguous()


def feats_to_img(feats, base_path, suffix="out", **kwargs):
    if feats is None:
        return
    feats = feats[0] if isinstance(feats, list) else feats
    import os

    base_path = os.path.join(base_path, suffix)
    if os.path.exists(base_path):
        pass
    else:
        # os.mkdir(base_path)
        os.makedirs(base_path)

    bs, c, h, w = feats.shape
    assert bs >= 1
    feats = feats[0].detach().cpu().numpy()

    for idx, feat in enumerate(feats):

        # print()
        plt.imsave(
            f"{base_path}/gray_scale_tensor{idx}.png",
            feat,
            cmap="gray",
        )


def feature_sampling(
    mlvl_feats,
    reference_voxel,
    pc_range,
    img_metas,
    img_depth=None,
    num_sweeps=1,
    num_cam=6,
    fp16_enabled=False,
    img_shape=None,
    post_rots=None,
    post_trans=None,
):
    lidar2img = []
    if not isinstance(img_metas, list):
        img_metas = [img_metas]

    img_depth = img_depth[None, ...]

    for img_meta in img_metas:
        lidar2img.append(img_meta["lidar2img"])

    # lidar2img = np.asarray(lidar2img)
    _l2c = []
    for trans_matrix in lidar2img:
        _l2c.append(torch.stack(trans_matrix))  # n 4 4
    lidar2img = torch.stack(_l2c).to(reference_voxel.device)

    if num_sweeps > 1:
        lidar2img = lidar2img[:, :, :num_sweeps]

    if lidar2img.shape[1] > num_cam:
        print(
            "WARNING: wanted num_cam {} but got {}".format(num_cam, lidar2img.shape[1])
        )
        num_cam = lidar2img.shape[1]

    # lidar2img = reference_voxel.new_tensor(lidar2img)  # (B, N, C, 4, 4)

    # Transfer to Point cloud range with X,Y,Z
    reference_voxel[..., 0:1] = (
        reference_voxel[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )  # X
    reference_voxel[..., 1:2] = (
        reference_voxel[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )  # Y
    reference_voxel[..., 2:3] = (
        reference_voxel[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )  # Z

    reference_aug = reference_voxel.clone()  # b x y z 3

    reference_voxel = torch.cat(
        (reference_voxel, torch.ones_like(reference_voxel[..., :1])), -1
    )  # b x y z 4
    reference_voxel = reference_voxel.flatten(1, 3)  # b, xyz, 4
    B, num_query = reference_voxel.size()[:2]

    if num_sweeps > 1:
        reference_voxel = reference_voxel.view(B, 1, 1, num_query, 4).repeat(
            1, num_cam, num_sweeps, 1, 1
        )
        reference_voxel = reference_voxel.reshape(
            B, num_cam * num_sweeps, num_query, 4, 1
        )
        lidar2img = lidar2img.view(B, num_cam, num_sweeps, 1, 4, 4)
        lidar2img = lidar2img.reshape(B, num_cam * num_sweeps, 1, 4, 4)
    else:
        reference_voxel = (
            reference_voxel.view(B, 1, num_query, 4)
            .repeat(1, num_cam, 1, 1)
            .unsqueeze(-1)
        )  # b n xyz 4 1
        lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)  # b n 1 4 4

    if fp16_enabled:
        lidar2img = lidar2img.half()
        reference_voxel = reference_voxel.half()
        img_depth = [_depth.half() for _depth in img_depth]
    else:
        img_depth = [_depth for _depth in img_depth]

    reference_voxel_cam = torch.matmul(
        lidar2img.to(torch.float), reference_voxel
    ).squeeze(
        -1
    )  # b n num_query(xyz) 4

    eps = 1e-5
    referenece_depth = reference_voxel_cam[..., 2:3].clone()  # Z-axis
    mask = referenece_depth > eps

    reference_voxel_cam = reference_voxel_cam[..., 0:2] / torch.maximum(
        reference_voxel_cam[..., 2:3],
        torch.ones_like(reference_voxel_cam[..., 2:3]) * eps,
    )  # X-Y-axis  -> b n xyz 2

    reference_voxel_cam = (
        torch.matmul(
            post_rots.unsqueeze(2)[..., :2, :2], reference_voxel_cam.unsqueeze(-1)
        ).squeeze(-1)
        + post_trans.unsqueeze(2)[..., :2]
    )

    # reference_voxel_cam = torch.nan_to_num(reference_voxel_cam)

    reference_voxel_cam[..., 0] /= img_shape[1]
    reference_voxel_cam[..., 1] /= img_shape[0]
    reference_voxel_cam = (reference_voxel_cam - 0.5) * 2
    # normalize depth
    if isinstance(img_depth, list):
        depth_dim = img_depth[0].shape[1]
    else:
        depth_dim = img_depth.shape[1]

    referenece_depth /= depth_dim
    referenece_depth = (referenece_depth - 0.5) * 2

    mask = (
        mask
        & (reference_voxel_cam[..., 0:1] > -1.0)
        & (reference_voxel_cam[..., 0:1] < 1.0)
        & (reference_voxel_cam[..., 1:2] > -1.0)
        & (reference_voxel_cam[..., 1:2] < 1.0)
        & (referenece_depth > -1.0)
        & (referenece_depth < 1.0)
    )

    mask = mask.view(B, num_cam * num_sweeps, 1, num_query, 1, 1).permute(
        0, 2, 3, 1, 4, 5
    )  # b 1 b xyz n 1 1
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_voxel_cam.view(
            B * N, num_query, 1, 2
        )  # 2 dims grid sampling
        sampled_feat = F.grid_sample(
            feat, reference_points_cam_lvl
        )  # (bn c h w) -> (b*n c num_query 1)  ; n for how many camera view is used
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(
            0, 2, 3, 1, 4
        )  # b, c,xyz,n,1
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)  # b, c,xyz,n,1,4
    sampled_feats = sampled_feats.view(
        B, C, num_query, num_cam * num_sweeps, 1, len(mlvl_feats)
    )  # b c xyz n 1 4

    # sample depth
    reference_points_cam = torch.cat(
        [reference_voxel_cam, referenece_depth], dim=-1
    )  # b n xyz 2 + 1 -> b n xyz 3
    reference_points_cam = reference_points_cam.view(
        B * num_cam * num_sweeps, 1, num_query, 1, 3
    )  # bn 1 xyz 1 3
    if isinstance(img_depth, list):
        sampled_depth = []
        for lvl, depth in enumerate(img_depth):
            depth = depth.unsqueeze(1)  # b*n,1,64,h,w
            depth = F.grid_sample(depth, reference_points_cam)  # bn 1 1 xyz 1
            depth = depth.view(B, num_cam * num_sweeps, 1, num_query, 1).permute(
                0, 2, 3, 1, 4
            )  # b 1 xyz n 1
            sampled_depth.append(depth)
        sampled_depth = torch.stack(sampled_depth, -1)  # b 1 xyz n 1 4
    else:
        img_depth = img_depth.unsqueeze(1)
        sampled_depth = F.grid_sample(img_depth, reference_points_cam)
        sampled_depth = sampled_depth.view(
            B, num_cam * num_sweeps, 1, num_query, 1
        ).permute(0, 2, 3, 1, 4)
        sampled_depth = sampled_depth.unsqueeze(-1)

    return sampled_feats, sampled_depth, mask, reference_aug


class ChannelOps(nn.Module):
    """Operation of Feats`s Channle
    Args:
        dim_in (int): the input feats channel.
        dim_out (int): the output feats channel, default, the out channel of feats is equal to 2* dim_in.
        padding (int): padding
        norm_cfg (bool): norm_cfg
        act_cfg (str): ReLU, Sigmoid, etc. act_cfg.
    Return:
        tensor: return the new calculation result of old_tensor
    """

    def __init__(
        self,
        dim_in=256,
        dim_out=None,
        kernel=1,
        padding=0,
        norm_cfg=None,
        act_cfg="ReLU",
    ):
        super(ChannelOps, self).__init__()
        dim_out = dim_in * 2 if dim_out is None else dim_out
        # self.conv2d = Conv2d(dim_in, out_channels=dim_out, kernel_size=1)
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01) if norm_cfg else None
        act_cfg = dict(type=act_cfg) if act_cfg else None
        self.conv2d = ConvModule(
            dim_in,
            dim_out,
            kernel_size=kernel,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        # self.conv2d[0].cuda()
        return self.conv2d(x)


class BiDirectionWeightFusion(nn.Module):
    def __init__(self, img_channels, radar_channels):
        super().__init__()

        self.img_channels = img_channels
        self.radar_channels = radar_channels

        weight_channels = img_channels + radar_channels
        self.img_bev_weight = nn.Conv2d(
            weight_channels, self.img_channels, 3, 1, padding=1
        )
        self.radar_bev_weight = nn.Conv2d(
            weight_channels, self.radar_channels, 3, 1, padding=1
        )
        # self.img_bev_weight = nn.Conv2d(weight_channels, 1, 3, 1, padding=1)
        # self.radar_bev_weight = nn.Conv2d(weight_channels, 1, 3, 1, padding=1)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def forward(self, img_bev, radar_bev):
        concat = torch.cat([img_bev, radar_bev], dim=1)

        # img_bev_weight = self.sig1(self.img_bev_weight(concat))
        # radar_bev_weight = self.sig2(self.radar_bev_weight(concat))

        img_bev_weight = torch.sigmoid(self.img_bev_weight(concat))
        radar_bev_weight = torch.sigmoid(self.radar_bev_weight(concat))

        assert img_bev_weight.size(2) == radar_bev_weight.size(2)

        img_bev = img_bev * img_bev_weight
        radar_bev = radar_bev * radar_bev_weight

        return img_bev, radar_bev


class Teacher_3DSparseTensor_Ops(nn.Module):
    def __init__(
        self, dim_in=[16, 32], layer_id=3, dim_out=None, kernel=1, padding=(1, 1, 1)
    ):
        super(Teacher_3DSparseTensor_Ops, self).__init__()

        self.dim_in = dim_in
        self.layer_id = layer_id

        # NOTE stilling changing !!!
        for i, in_channel in enumerate(dim_in):
            if len(dim_in) == 1:
                teacher_conv_modify = spconv.SparseSequential(
                    SparseConv3d(
                        dim_in[0], dim_in[0], 3, padding=padding, bias=False
                    ),  # [200, 150, 5] -> [200, 150, 2]   1/16
                    build_norm_layer(
                        dict(type="BN1d", eps=1e-3, momentum=0.01), dim_in[0]
                    )[1],
                    nn.ReLU(),
                )
                self.__setattr__(f"conv{layer_id[0]}_modify", teacher_conv_modify)
            else:
                teacher_conv_modify = spconv.SparseSequential(
                    SparseConv3d(
                        in_channel, in_channel, 3, padding=padding, bias=False
                    ),  # [200, 150, 5] -> [200, 150, 2]   1/16
                    build_norm_layer(
                        dict(type="BN1d", eps=1e-3, momentum=0.01), in_channel
                    )[1],
                    nn.ReLU(),
                )

                self.__setattr__(f"conv{i}_modify", teacher_conv_modify)

        self.conv3d_modify = ConvModule(
            128, 64, kernel_size=1, stride=(2, 1, 1), conv_cfg=dict(type="Conv3d")
        )

        # x= nn.Conv3d(
        #     128, 64, kernel_size=1, stride=(1, 1, 1))

    def forward(self, x):
        ret_x = []
        if isinstance(x, dict):
            for i in range(len(self.dim_in)):
                ret_x.append(self.__getattr__(f"conv{i}_modify")(x[f"conv{i}"]))
        elif isinstance(x, list):
            for i in range(len(self.dim_in)):
                # print(colored(x[i].dense().shape, 'red'))
                ret_x[f"conv{i}"] = (
                    self.__getattr__(f"conv{i}_modify")(x[i]).dense().clone()
                )
        else:
            # NOTE The 3D low voxel distillation method temporarily configured for fastbev requires too many changes and can only be changed as needed.
            modifty_3d_voxel_feats = self.conv3d_modify(x.clone())
            # print(f"===={modifty_3d_voxel_feats.shape}===")
            if modifty_3d_voxel_feats.size(2) == 1:
                modifty_3d_voxel_feats = F.interpolate(
                    modifty_3d_voxel_feats,
                    (4, modifty_3d_voxel_feats.size(3), modifty_3d_voxel_feats.size(4)),
                    mode="trilinear",
                )
                # modifty_3d_voxel_feats=nn.ConvTranspose3d(64,64)
            # print(f"===={modifty_3d_voxel_feats.shape}===")
            ret_x.append(modifty_3d_voxel_feats)

        return ret_x


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.activation = nn.Sigmoid()
        # self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.activation(x)


class MS_CAM(nn.Module):

    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # return out

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DualBevAdaptiveFusion(nn.Module):
    """DualBevAdaptiveFusion
    Args:
        stu_bev_channel (int): channel num of stu_distill_feats.
        duplicated_bev_channel (int): channel num of dual_encode_feats.
        padding (int): padding for fit conv out.
        norm_cfg (bool): normalization config.
        act_cfg (str): Optional params in [ReLU, Sigmoid, etc].
        adaptive_type (str): Optional params in [se-like or weight-fusion]
    Return:
        bev_feats (tensor): fusion bev feats
    """

    def __init__(
        self,
        stu_bev_channel,
        duplicated_bev_channel,
        padding=1,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU", inplace=True),
        reduce="cba",
        adaptive_type="se-like",
    ):
        super(DualBevAdaptiveFusion, self).__init__()
        self.adaptive_type = adaptive_type

        if adaptive_type == "weight-fusion":
            self.weight_gen = MS_CAM(128, 64)
        elif adaptive_type == "se-like":
            if reduce == "cba":
                self.reduce_channel = ConvModule(
                    stu_bev_channel + duplicated_bev_channel,
                    stu_bev_channel,
                    kernel_size=3,
                    padding=padding,
                    norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
                    act_cfg=act_cfg,
                )
            elif reduce == "conv2d":
                self.reduce_channel = nn.Conv2d(
                    stu_bev_channel + duplicated_bev_channel,
                    stu_bev_channel,
                    kernel_size=1,
                    stride=1,
                )
            else:
                pass

            self.attention_learning = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    stu_bev_channel,
                    stu_bev_channel,
                    kernel_size=1,
                    stride=1,
                ),
                nn.Sigmoid(),
            )
            self.attention_spatial = SpatialAttention()
            self.residual_encoder = nn.Sequential(
                ConvModule(
                    stu_bev_channel,
                    stu_bev_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="BN"),
                    inplace=True,
                ),
                ConvModule(
                    stu_bev_channel,
                    stu_bev_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="BN"),
                    act_cfg=None,
                ),
            )
            # self.out_conv = ConvModule(
            #     in_channels=stu_bev_channel,
            #     out_channels=stu_bev_channel,
            #     kernel_size=3,
            #     padding=1,
            #     stride=1,
            #     norm_cfg=dict(type="BN2d"),
            # )
        else:
            self.bi_direction_fusion = BiDirectionWeightFusion(
                stu_bev_channel, duplicated_bev_channel
            )
            self.out_res_conv = ResBlock(
                stu_bev_channel + duplicated_bev_channel,
                stu_bev_channel,
                downsample=ConvModule(
                    in_channels=stu_bev_channel + duplicated_bev_channel,
                    out_channels=stu_bev_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=None,
                    act_cfg=None,
                ),
            )

    def se_like(self, stu_bev_feats, duplicated_bev_feats):

        # residual = stu_bev_feats

        # hybird_spatial = self.attention_spatial(duplicated_bev_feats)
        # lss_hybird_sampliing = stu_bev_feats * hybird_spatial

        # lss_spatial = self.attention_spatial(stu_bev_feats)
        # hybird_lss_sampling = duplicated_bev_feats * lss_spatial
        # bev_feats = torch.cat([lss_hybird_sampliing, hybird_lss_sampling], dim=1)

        bev_feats = torch.cat([stu_bev_feats, duplicated_bev_feats], dim=1)

        attention_spatial = self.attention_spatial(bev_feats)
        # lss_hybird_sampliing = stu_bev_feats * attention_spatial
        hybird_lss_sampling = duplicated_bev_feats * attention_spatial

        bev_feats = torch.cat([stu_bev_feats, hybird_lss_sampling], dim=1)
        # bev_feats = self.reduce_channel(bev_feats)
        # attention_learning = self.attention_learning(bev_feats)
        # bev_feats = bev_feats * attention_learning

        # bev_feats += residual
        # bev_feats = self.out_conv(bev_feats)

        return bev_feats

    def weight_fusion(self, stu_bev_feats, duplicated_bev_feats):
        assert stu_bev_feats.size(1) == duplicated_bev_feats.size(1)
        bev_feats = torch.cat([stu_bev_feats, duplicated_bev_feats], dim=1)

        fusion_weight = self.weight_gen(bev_feats)
        bev_feats = (
            fusion_weight * stu_bev_feats + (1 - fusion_weight) * duplicated_bev_feats
        )
        return bev_feats

    def forward(self, stu_bev_feats, duplicated_bev_feats):
        # bev_feats = torch.cat([stu_bev_feats, duplicated_bev_feats], dim=1)
        # bev_feats = self.reduce_channel(bev_feats)

        if self.adaptive_type == "se-like":
            bev_feats = self.se_like(
                stu_bev_feats=stu_bev_feats, duplicated_bev_feats=duplicated_bev_feats
            )
        elif self.adaptive_type == "weight-fusion":
            bev_feats = self.weight_fusion(
                stu_bev_feats=stu_bev_feats, duplicated_bev_feats=duplicated_bev_feats
            )
        else:
            bev_feats, duplicated_bev_feats = self.bi_direction_fusion(
                stu_bev_feats, duplicated_bev_feats
            )
            bev_feats = self.out_res_conv(
                torch.cat([bev_feats, duplicated_bev_feats], dim=1)
            )

        return bev_feats
