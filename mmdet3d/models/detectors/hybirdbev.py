import torch
from torch.nn import functional as F
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmcv.cnn import build_norm_layer, Conv2d, Conv3d
from mmcv.cnn.bricks import ConvModule
from mmdet.models import build_detector
from mmcv.runner import force_fp32
from .. import builder
from ..builder import DETECTORS, build_loss
from .centerpoint import CenterPoint
from termcolor import colored
import torch.nn as nn
from mmdet3d.models.losses import kd_loss as IKDLOSS
import copy
from einops import rearrange
from mmdet3d.models.necks.hybird_encoder import (
    HybirdEncoder,
    DualBevAdaptiveFusion,
    ResBlock,
    ChannelOps,
    Teacher_3DSparseTensor_Ops,
    feats_to_img,
    SCConv,
)


@DETECTORS.register_module()
class HybirdBEV(CenterPoint):

    def __init__(
        self,
        is_distill,
        img_view_transformer,
        img_bev_encoder_backbone=None,
        img_bev_encoder_neck=None,
        x_sample_num=16,
        y_sample_num=16,
        embed_channels=[256, 512],
        inter_keypoint_weight=0.0,
        enlarge_width=-1,
        inner_depth_weight=0.0,
        inter_channel_weight=0.0,
        distill_config=None,
        teacher_config_file=None,
        teacher_pretrained=None,
        student_img_pretrained=None,
        pre_process=None,
        hybird_encoder=None,
        **kwargs,
    ):
        super(HybirdBEV, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(
                img_bev_encoder_backbone
            )
        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        self.hybird_encoder = builder.build_neck(hybird_encoder)
        # self.hybird_conv = ResBlock(128, 64)
        self.pre_process = builder.build_neck(pre_process) if pre_process else None

        self.inter_keypoint_weight = inter_keypoint_weight
        self.inter_channel_weight = inter_channel_weight
        self.inner_depth_weight = inner_depth_weight
        self.enlarge_width = enlarge_width
        self.x_sample_num = x_sample_num
        self.y_sample_num = y_sample_num
        self.embed_channels = embed_channels

        self.is_distill = is_distill
        self.distill_config = distill_config
        self.low_config = distill_config.low_level_distill
        self.dual_encode_config = self.low_config.dual_bev_encode
        self.dual_encode_config.only_training = False

        # if isinstance(
        #     self.dual_encode_config.img_bev_encoder_neck_2.out_channels, list
        # ):
        #     self.dual_encode_config.duplicated_bev_channel = sum(
        #         self.dual_encode_config.img_bev_encoder_neck_2.out_channels
        #     )
        # else:
        #     self.dual_encode_config.duplicated_bev_channel = (
        #         self.dual_encode_config.img_bev_encoder_neck_2.out_channels
        #     )

        self.high_config = distill_config.high_level_distill
        self.ms_distill_config = distill_config.multi_scale_distill

        self.is_low_distill = self.low_config.is_low_level_distill
        self.is_dual_encode = self.low_config.is_dual_bev_encode
        self.is_high_distill = self.high_config.is_high_level_distill
        self.is_ms_distill = self.ms_distill_config.is_multi_scale_distill

        self.student_img_pretrained = student_img_pretrained

        # self.conv = nn.Sequential(
        #     nn.Conv2d(
        #         64,
        #         embed_channels[0],
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     build_norm_layer(dict(type="BN"), embed_channels[0], postfix=0)[1],
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(
        #         embed_channels[0],
        #         embed_channels[0],
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     build_norm_layer(dict(type="BN"), embed_channels[0], postfix=0)[1],
        #     nn.ReLU(inplace=True),
        # )

        ###############################################
        #          load the teacher net model         #
        ###############################################
        # load the teacher model
        teacher_cfg = Config.fromfile(teacher_config_file)
        self.teacher_cfg = teacher_cfg
        self.teacher_model = build_detector(
            teacher_cfg.model,
            train_cfg=teacher_cfg.get("train_cfg"),
            test_cfg=teacher_cfg.get("test_cfg"),
        )
        self.teacher_pretrained = teacher_pretrained
        self.load_teacher_ckpt(teacher_pretrained)

        if self.distill_config.freeze_teacher:
            # self.frozen_pretrained(self.teacher_model)
            for params in self.teacher_model.parameters():
                params.requires_grad = False
            self.teacher_model.eval()

        ###############################################
        #    load low distill channel align config    #
        ###############################################
        # low level kd disitll module, adpat fit channel size
        if self.is_low_distill and self.low_config.low_level_distill_type == "2d":
            if self.low_config.low_level_distill_loss.type == "CriterionCWD_KD":
                self.stu_low_level_feats_channel_expander = Conv2d(
                    64, embed_channels[0], kernel_size=1, stride=1
                )
            elif self.low_config.loss_type == "bev-v1":
                self.stu_low_level_feats_channel_expander = Conv2d(
                    64, embed_channels[0], kernel_size=1, stride=1
                )
            elif self.low_config.loss_type == "bev-v2":
                self.stu_low_level_feats_channel_expander = ChannelOps(
                    64, embed_channels[0]
                )

        if (
            self.is_dual_encode
            and distill_config.low_level_distill.dual_bev_encode.get(
                "img_bev_encoder_backbone_2", None
            )
        ):
            self.img_bev_encoder_backbone_2 = builder.build_backbone(
                distill_config.low_level_distill.dual_bev_encode.img_bev_encoder_backbone_2
            )
            self.img_bev_encoder_neck_2 = builder.build_neck(
                distill_config.low_level_distill.dual_bev_encode.img_bev_encoder_neck_2
            )

        if (
            self.is_dual_encode
            and self.dual_encode_config.dual_bev_encode_feats_fusion == "adaptive"
        ):
            # get the channel size
            student_bev_channel = self.dual_encode_config.student_bev_channel[0]
            duplicated_bev_channel = (
                self.dual_encode_config.duplicated_bev_channel
                if self.dual_encode_config.duplicated_bev_channel
                else 256
            )

            # config the dual bev encode
            self.dual_bev_adaptive_fusion = DualBevAdaptiveFusion(
                student_bev_channel,
                duplicated_bev_channel,
                adaptive_type=self.dual_encode_config.adaptive_type,
            )

        if self.distill_config.get("voxel_feats_distill", None):
            self.voxel_feats_distill = build_loss(
                self.distill_config.voxel_feats_distill
            )

        # NOTE config 3d voxel feats distillation for multi-scale feats
        if self.low_config.low_level_distill_type == "3d":
            self.low_ms_voxel_feats_modify = Teacher_3DSparseTensor_Ops(
                self.low_config.s_conv_modify.in_channel,
                self.low_config.s_conv_modify.layer_id,
            )

        if len(embed_channels) == 2:
            self.embed = nn.Conv2d(
                embed_channels[0] * 3, embed_channels[1], kernel_size=1, padding=0
            )
            self.embed_1 = nn.Conv2d(
                embed_channels[0] // 2, embed_channels[1] // 2, kernel_size=1, padding=0
            )

        self.img_bev_clone = ConvModule(64, 64, kernel_size=1, stride=1)

        if distill_config.get("lock_head"):
            print(colored("lock det head weight parameters!", "yellow"))
            if self.pts_bbox_head:
                for param in self.pts_bbox_head.parameters():
                    param.requires_grad = False
                self.pts_bbox_head.eval()

        if self.is_low_distill:
            self.low_level_bev_feats_kd_loss = build_loss(
                self.low_config.low_level_distill_loss
            )

        if "moco_kd_loss_yes" in distill_config:  # moco_kd_loss_yes or moco_kd_loss_no
            self.low_level_moco_kd_loss = build_loss(distill_config.moco_kd_loss)

        if self.is_ms_distill:
            self.multi_scale_kd_distill_loss = build_loss(
                self.ms_distill_config.multi_scale_distill_loss
            )

        if self.is_high_distill:
            self.hightlevel_bev_kd_loss = build_loss(
                self.high_config.high_level_distill_loss
            )
        self.hybird_latent = nn.Sequential(
            ConvModule(**hybird_encoder.conv_dict),
            ConvModule(**hybird_encoder.conv_dict),
            ConvModule(**hybird_encoder.conv_dict),
        )
        self.hybird_recude = nn.Sequential(
            ConvModule(
                128 * hybird_encoder.num_points,
                (128 * hybird_encoder.num_points) // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ),
            ConvModule(
                (128 * hybird_encoder.num_points) // 2,
                128 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ),
            ConvModule(
                64,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ),
            # SCConv(64, 64, 4, norm_layer=nn.BatchNorm2d),
        )
        self.residual_process = ResBlock(64, 64)

        self.reduce_channel = ConvModule(
            128,
            64,
            3,
            1,
            1,
            norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
            act_cfg=dict(type="ReLU"),
        )

        self.distill_feats = {}

    def init_weights(self):
        from mmdet3d.models.utils.self_print import print2file

        teacher_ckpt_load = torch.load(
            self.teacher_pretrained,
            map_location="cuda:{}".format(torch.cuda.current_device()),
        )["state_dict"]

        if self.distill_config.stu_load_tea_head:
            print(
                colored(
                    "Loaded pretrained teacher model from: {}".format(
                        self.teacher_pretrained
                    ),
                    "red",
                )
            )

            dict_state_load = {
                _key.replace("pts_bbox_head.", ""): teacher_ckpt_load[_key]
                for _key in teacher_ckpt_load
                if "pts_bbox_head" in _key
            }

            if self.dual_encode_config.dual_bev_encode_feats_fusion == "cat":
                dict_state_load_new = dict_state_load
            else:
                # 检测头的第一层的通道reduce不匹配，直接在k-v权重中裁剪掉
                dict_state_load_new = {}
                for key, value in dict_state_load.items():
                    if "shared_conv" in key:
                        pass
                    else:
                        dict_state_load_new.update({key: value})

            # print2file({'dict_state_load': dict_state_load}, 'dict_state_load')

            print(
                colored(
                    "Loaded pretrained for student pts_bbox_head from pretrained teacher pts_bbox_head!",
                    "red",
                )
            )
            self.pts_bbox_head.load_state_dict(dict_state_load_new, strict=False)
            assert len(dict_state_load) > 0

    def load_teacher_ckpt(self, teacher_ckpt):
        checkpoint = load_checkpoint(
            self.teacher_model, teacher_ckpt, map_location="cpu"
        )

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        # print(f"===ms img feats size: {len(x)}==={x[1].shape}==={x[0].shape}==")
        if self.with_img_neck:
            x = self.img_neck(x)

        if type(x) == list or type(x) == tuple:
            x = x[0]

        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    def bev_encoder(self, x, ms_distill=False):
        x = self.img_bev_encoder_backbone(x)

        if ms_distill:
            ms_x = x
            x = self.img_bev_encoder_neck(x)
            return x, ms_x
        else:
            x = self.img_bev_encoder_neck(x)

            return x

    def wrap_bev_encoder_with_ms_distill(self, x, ret_feats, is_multi_scale_distill):
        # 高层特征密集多尺度特征处理
        if is_multi_scale_distill:
            x, ms_x = self.bev_encoder(x, ms_distill=True)
            ret_feats.update({"multi_scale_feats": ms_x})
        else:
            x = self.bev_encoder(x)

        return x

    def hybird_process(
        self,
        img_feat,
        depth,
        img_metas,
        bs,
        process="cat",
        post_rots=None,
        post_trans=None,
    ):
        hybird_feats = self.hybird_encoder(
            img_feat,
            img_depth=depth,
            img_metas=img_metas,
            batch_size=bs,
            img_shape=self.img_view_transformer.data_config.input_size,
            post_rots=post_rots,
            post_trans=post_trans,
        )
        hybird_feats = self.hybird_latent(hybird_feats.to(torch.float32))
        self.distill_feats["stu_voxel_feats"] = hybird_feats

        if "cat" in process:
            hybird_feats = hybird_feats.unbind(dim=2)
            hybird_feats = torch.cat(hybird_feats, dim=1)
        elif "sum" in process:
            hybird_feats = torch.sum(hybird_feats, dim=2)
        hybird_feats = self.hybird_recude(hybird_feats)  # B, 64, H, W

        # hybird_feats = self.hybird_conv(hybird_feats.to(torch.float32))

        return hybird_feats

    def dual_bev_encoder_fusion(self, x, dual_low_level_feats, fusion="adaptive"):

        if self.is_dual_encode:
            if fusion == "cat":
                x = torch.cat([x, dual_low_level_feats], dim=1)
            elif fusion == "sum":
                x = sum([x, dual_low_level_feats])
            elif fusion == "adaptive":
                x = self.dual_bev_adaptive_fusion(x, dual_low_level_feats)
            return x

    def extract_pts_feat(self, pts, img_feats, img_metas):

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):

        low_config = self.distill_config.low_level_distill
        low_feat_type = low_config.low_level_distill_type
        low_feat_dual = low_config.is_dual_bev_encode

        x = self.image_encoder(img[0])
        img_feat = x
        bs = x.size(0)

        x, depth, stu_low_level_3d = self.img_view_transformer([x] + img[1:])
        residuial_x = x
        # feats_to_img(feats=x, base_path=self.base_path, suffix="lss")
        # hybird encoder
        hybird_bev = self.hybird_process(
            img_feat=img_feat,
            depth=depth,
            img_metas=img_metas,
            bs=bs,
            post_rots=img[4],
            post_trans=img[5],
        )  # c 64
        # feats_to_img(feats=hybird_bev, base_path=self.base_path, suffix="hybird")

        x = self.dual_bev_encoder_fusion(
            x,
            hybird_bev,
            self.dual_encode_config.dual_bev_encode_feats_fusion,
        )
        if self.pre_process:
            x = self.pre_process(x)[0]
        if x.size(1) != 64:
            x = self.reduce_channel(x)
        ret_feats = {"stu_voxel_feats": stu_low_level_3d, "low_level_feats": x}

        x = self.wrap_bev_encoder_with_ms_distill(x, ret_feats, self.is_ms_distill)[0]

        return x, depth, ret_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""

        img_feats, depth, feats_dict = self.extract_img_feat(img, img_metas)

        return img_feats, depth, feats_dict

    def get_depth_loss(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (
            (~(depth_gt == 0))
            .reshape(B, N, 1, H, W)
            .expand(B, N, self.img_view_transformer.D, H, W)
        )
        depth_gt = (
            depth_gt - self.img_view_transformer.grid_config["dbound"][0]
        ) / self.img_view_transformer.grid_config["dbound"][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0, self.img_view_transformer.D).to(
            torch.long
        )
        depth_gt_logit = F.one_hot(
            depth_gt.reshape(-1), num_classes=self.img_view_transformer.D
        )
        depth_gt_logit = (
            depth_gt_logit.reshape(B, N, H, W, self.img_view_transformer.D)
            .permute(0, 1, 4, 2, 3)
            .to(torch.float32)
        )
        depth = depth.sigmoid().view(B, N, self.img_view_transformer.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit, weight=loss_weight)
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    def get_gt_sample_grid(self, corner_points2d):
        dH_x, dH_y = corner_points2d[0] - corner_points2d[1]
        dW_x, dW_y = corner_points2d[0] - corner_points2d[2]
        raw_grid_x = (
            torch.linspace(
                corner_points2d[0][0], corner_points2d[1][0], self.x_sample_num
            )
            .view(1, -1)
            .repeat(self.y_sample_num, 1)
        )
        raw_grid_y = (
            torch.linspace(
                corner_points2d[0][1], corner_points2d[2][1], self.y_sample_num
            )
            .view(-1, 1)
            .repeat(1, self.x_sample_num)
        )
        raw_grid = torch.cat((raw_grid_x.unsqueeze(2), raw_grid_y.unsqueeze(2)), dim=2)
        raw_grid_x_offset = (
            torch.linspace(0, -dW_x, self.x_sample_num)
            .view(-1, 1)
            .repeat(1, self.y_sample_num)
        )
        raw_grid_y_offset = (
            torch.linspace(0, -dH_y, self.y_sample_num)
            .view(1, -1)
            .repeat(self.x_sample_num, 1)
        )
        raw_grid_offset = torch.cat(
            (raw_grid_x_offset.unsqueeze(2), raw_grid_y_offset.unsqueeze(2)), dim=2
        )
        grid = raw_grid + raw_grid_offset  # X_sample,Y_sample,2
        grid[:, :, 0] = torch.clip(
            (
                (
                    grid[:, :, 0]
                    - (
                        self.img_view_transformer.bx[0].to(grid.device)
                        - self.img_view_transformer.dx[0].to(grid.device) / 2.0
                    )
                )
                / self.img_view_transformer.dx[0].to(grid.device)
                / (self.img_view_transformer.nx[0].to(grid.device) - 1)
            )
            * 2.0
            - 1.0,
            min=-1.0,
            max=1.0,
        )
        grid[:, :, 1] = torch.clip(
            (
                (
                    grid[:, :, 1]
                    - (
                        self.img_view_transformer.bx[1].to(grid.device)
                        - self.img_view_transformer.dx[1].to(grid.device) / 2.0
                    )
                )
                / self.img_view_transformer.dx[1].to(grid.device)
                / (self.img_view_transformer.nx[1].to(grid.device) - 1)
            )
            * 2.0
            - 1.0,
            min=-1.0,
            max=1.0,
        )

        return grid.unsqueeze(0)

    def get_inner_feat(self, gt_bboxes_3d, img_feats, pts_feats):
        device = img_feats.device
        dtype = img_feats[0].dtype

        img_feats_sampled_list = []
        pts_feats_sampled_list = []

        for sample_ind in torch.arange(len(gt_bboxes_3d)):
            img_feat = img_feats[sample_ind].unsqueeze(0)  # 1,C,H,W
            pts_feat = pts_feats[sample_ind].unsqueeze(0)  # 1,C,H,W

            bbox_num, corner_num, point_num = gt_bboxes_3d[sample_ind].corners.shape

            for bbox_ind in torch.arange(bbox_num):
                if self.enlarge_width > 0:
                    gt_sample_grid = self.get_gt_sample_grid(
                        gt_bboxes_3d[sample_ind]
                        .enlarged_box(self.enlarge_width)
                        .corners[bbox_ind][[0, 2, 4, 6], :-1]
                    ).to(device)
                else:
                    gt_sample_grid = self.get_gt_sample_grid(
                        gt_bboxes_3d[sample_ind].corners[bbox_ind][[0, 2, 4, 6], :-1]
                    ).to(
                        device
                    )  # 1,sample_y,sample_x,2

                # 'bilinear')) #all_bbox_num,C,y_sample,x_sample
                img_feats_sampled_list.append(
                    F.grid_sample(
                        img_feat,
                        grid=gt_sample_grid,
                        align_corners=False,
                        mode="bilinear",
                    )
                )
                # 'bilinear')) #all_bbox_num,C,y_sample,x_sample
                pts_feats_sampled_list.append(
                    F.grid_sample(
                        pts_feat,
                        grid=gt_sample_grid,
                        align_corners=False,
                        mode="bilinear",
                    )
                )

        return torch.cat(img_feats_sampled_list, dim=0), torch.cat(
            pts_feats_sampled_list, dim=0
        )

    def get_inner_depth_loss(self, fg_gt, depth_gt, depth):
        B, N, H, W = fg_gt.shape
        depth_prob = depth.softmax(dim=1)  # B*N,D,H,W
        discrete_depth = (
            torch.arange(
                *self.img_view_transformer.grid_config["dbound"],
                dtype=torch.float,
                device=depth_prob.device,
            )
            .view(1, -1, 1, 1)
            .expand(B * N, -1, H, W)
        )
        depth_estimate = torch.sum(depth_prob * discrete_depth, dim=1)  # B*N,H,W
        fg_gt = fg_gt.reshape(B * N, H, W)
        depth_gt = depth_gt.reshape(B * N, H, W)
        pos_mask = fg_gt > 0
        num_pos = torch.clip(pos_mask.sum(), min=1.0)

        fg_gt = F.one_hot(fg_gt.long()).permute(0, 3, 1, 2)[:, 1:, :, :]

        depth_estimate_fg = fg_gt * (depth_estimate.unsqueeze(1))  # B*N,num_bbox,H,W
        depth_fg_gt = fg_gt * (depth_gt.unsqueeze(1))  # B*N,num_bbox,H,W

        depth_dis = torch.abs(depth_estimate_fg - depth_fg_gt) + 99 * (1 - fg_gt)
        depth_min_dis_ind = depth_dis.view(B * N, -1, H * W).argmin(-1, keepdim=True)

        depth_estimate_fg_min = fg_gt * (
            depth_estimate_fg.view(B * N, -1, H * W).gather(-1, depth_min_dis_ind)
        ).view(B * N, -1, 1, 1)
        depth_fg_gt_min = fg_gt * (
            depth_fg_gt.view(B * N, -1, H * W).gather(-1, depth_min_dis_ind)
        ).view(B * N, -1, 1, 1)

        diff_pred = depth_estimate - depth_estimate_fg_min.sum(1).detach()  # B*N,H,W
        diff_gt = depth_gt - depth_fg_gt_min.sum(1)
        loss_inner_depth = (
            torch.sum(
                F.mse_loss(diff_pred, diff_gt.detach(), reduction="none") * pos_mask
            )
            / num_pos
            * self.inner_depth_weight
        )

        return loss_inner_depth

    def get_inter_keypoint_loss(self, img_feats_kd, pts_feats_kd):
        C_img = img_feats_kd.shape[1]
        C_pts = pts_feats_kd.shape[1]
        N = self.x_sample_num * self.y_sample_num

        img_feats_kd = (
            img_feats_kd.view(-1, C_img, N)
            .permute(0, 2, 1)
            .matmul(img_feats_kd.view(-1, C_img, N))
        )  # -1,N,N
        pts_feats_kd = (
            pts_feats_kd.view(-1, C_pts, N)
            .permute(0, 2, 1)
            .matmul(pts_feats_kd.view(-1, C_pts, N))
        )

        img_feats_kd = F.normalize(img_feats_kd, dim=2)
        pts_feats_kd = F.normalize(pts_feats_kd, dim=2)

        loss_inter_keypoint = F.mse_loss(img_feats_kd, pts_feats_kd, reduction="none")
        loss_inter_keypoint = loss_inter_keypoint.sum(-1)
        loss_inter_keypoint = loss_inter_keypoint.mean()
        loss_inter_keypoint = self.inter_keypoint_weight * loss_inter_keypoint
        return loss_inter_keypoint

    def get_inter_channel_loss(self, img_feats_kd, pts_feats_kd):
        if img_feats_kd.size(1) == 128:
            img_feats_kd = self.embed_1(img_feats_kd)
        else:
            img_feats_kd = self.embed(img_feats_kd)
        C_img = img_feats_kd.shape[1]
        C_pts = pts_feats_kd.shape[1]
        N = self.x_sample_num * self.y_sample_num

        img_feats_kd = img_feats_kd.view(-1, C_img, N).matmul(
            img_feats_kd.view(-1, C_img, N).permute(0, 2, 1)
        )  # -1,N,N
        pts_feats_kd = pts_feats_kd.view(-1, C_pts, N).matmul(
            pts_feats_kd.view(-1, C_pts, N).permute(0, 2, 1)
        )

        img_feats_kd = F.normalize(img_feats_kd, dim=2)
        pts_feats_kd = F.normalize(pts_feats_kd, dim=2)

        loss_inter_channel = F.mse_loss(img_feats_kd, pts_feats_kd, reduction="none")
        loss_inter_channel = loss_inter_channel.sum(-1)
        loss_inter_channel = loss_inter_channel.mean()
        loss_inter_channel = self.inter_channel_weight * loss_inter_channel
        return loss_inter_channel

    def get_teacher_kd(
        self,
        points,
        img_inputs,
        img_metas,
        gt_bboxes_3d,
    ):

        def wrap_teacher_distill_feats(ret_feats):

            if self.low_config.low_level_distill_type == "2d":
                assert isinstance(ret_feats[-1], dict)
                return (
                    ret_feats[1],
                    ret_feats[-1]["low_bev_feats"],
                    ret_feats[-1]["high_ms_feats"],
                )

            elif self.low_config.low_level_distill_type == "3d":
                # get low ms voxel feats
                low_ms_feats = ret_feats[-1]["low_ms_voxel_feats"]
                low_ms_voxel_feats = self.low_ms_voxel_feats_modify(
                    low_ms_feats
                )  # list[tensor of ms voxel feats]

                return ret_feats[1], low_ms_voxel_feats, ret_feats[-1]["high_ms_feats"]

        # ret_feats: img_feats+ pts_feats + {'low_ms_voxel_feats', 'low_bev_feats', 'high_ms_feats'}
        ret_feats = self.teacher_model.extract_feat(points, img_inputs, img_metas)

        (
            pts_feats,
            teacher_low_feats,
            teacher_high_ms_feats,
        ) = wrap_teacher_distill_feats(ret_feats=ret_feats)

        return pts_feats, teacher_low_feats, teacher_high_ms_feats, ret_feats

    def get_low_level_feats_distill_2d(
        self, stu_feats, tea_feats, loss_type="bev-v2", gt_bboxes_3d=None
    ):

        stu_feats = stu_feats.get("low_level_feats", None)
        bsz, student_C, student_H, student_W = stu_feats.shape
        _, teacher_C, teacher_H, teacher_W = tea_feats.shape
        assert teacher_C == self.embed_channels[0]

        if loss_type == "moco":
            stu_feats = rearrange(stu_feats["low_level_feats"], "b c h w -> (b h w) c")
            tea_feats = rearrange(tea_feats, "b c h w -> (b h w) c")
            loss_low_level_bev_kd = self.low_level_moco_kd_loss(stu_feats, tea_feats)

        elif loss_type == "bev-v1":
            stu_feats = (
                self.stu_low_level_feats_channel_expander(stu_feats)
                if student_C == 64
                else stu_feats
            )
            loss_low_level_bev_kd = self.low_level_bev_feats_kd_loss(
                stu_feats, tea_feats
            )

        else:

            self.inter_keypoint_weight = 100.0
            self.inter_channel_weight = 10.0
            loss_low_level_bev_kd = 0

            if self.inter_keypoint_weight > 0 or self.inter_channel_weight > 0:
                student_bev, teacher_bev = self.get_inner_feat(
                    gt_bboxes_3d, stu_feats, tea_feats
                )

            if self.inter_keypoint_weight > 0:
                loss_inter_keypoint = self.get_inter_keypoint_loss(
                    student_bev, teacher_bev
                )
                loss_low_level_bev_kd += loss_inter_keypoint

            if self.inter_channel_weight > 0:
                loss_inter_channel = self.get_inter_channel_loss(
                    student_bev, teacher_bev
                )
                loss_low_level_bev_kd += loss_inter_channel

        return loss_low_level_bev_kd

    def get_low_level_feats_distill(
        self, stu_feats, tea_feats, loss_type="bev-v2", gt_boxes_3d=None
    ):
        if self.low_config.low_level_distill_type == "2d":
            loss_low_level_kd = self.get_low_level_feats_distill_2d(
                stu_feats=stu_feats,
                tea_feats=tea_feats,
                loss_type=loss_type,
                gt_bboxes_3d=gt_boxes_3d,
            )

        return loss_low_level_kd

    def get_channel_agnostic_distill(self, student_feats, teacher_feats):
        bsz, C_img, stu_num_query = rearrange(
            student_feats, "b c h w -> b c (h w)"
        ).shape
        # C_img = student_feats.shape[1]

        _, C_pts, tea_num_query = rearrange(teacher_feats, "b c h w -> b c (h w)").shape
        # C_pts = teacher_feats.shape[1]

        # N = self.x_sample_num * self.y_sample_num

        student_feats = (
            student_feats.view(-1, C_img, stu_num_query).permute(0, 2, 1)
            # grid_sample-> h*w -> y_sample_num*x_sample_num -> view(b, c, y_sample_num*x_sample_num)
            .matmul(student_feats.view(-1, C_img, stu_num_query))
        )  # -1,N,N
        teacher_feats = (
            teacher_feats.view(-1, C_pts, tea_num_query)
            .permute(0, 2, 1)
            .matmul(teacher_feats.view(-1, C_pts, tea_num_query))
        )

        student_feats = F.normalize(student_feats, dim=2)
        teacher_feats = F.normalize(teacher_feats, dim=2)

        loss_channel_agnostic = F.mse_loss(
            student_feats, teacher_feats, reduction="none"
        )
        loss_channel_agnostic = loss_channel_agnostic.sum(-1)
        loss_channel_agnostic = loss_channel_agnostic.mean()
        loss_channel_agnostic = 20 * loss_channel_agnostic
        return loss_channel_agnostic

    def get_multi_scale_dense_distill(self, ms_stu_list, ms_tea_list):
        assert len(ms_stu_list) > 1
        assert len(ms_tea_list) > 1

        unfied_feats_tea = []
        for i, _ in enumerate(ms_tea_list):
            _ms_stu, _ms_tea = self.unify_feat_size(ms_stu_list[i], ms_tea_list[i])
            unfied_feats_tea.append(_ms_tea)

        loss_ms_kd = self.multi_scale_kd_distill_loss(ms_stu_list[:2], unfied_feats_tea)
        return loss_ms_kd

    def get_high_level_bev_kd_loss(self, student_bev, teacher_bev, gt_bboxes_3d):
        # print(            f"==student : ={student_bev.shape}==teacher:={teacher_bev.shape}===")
        student_bev_channel = student_bev.shape[1]
        teacher_bev_channel = teacher_bev.shape[1]

        loss_highlevel_bev_list = []

        if student_bev_channel == teacher_bev_channel:
            loss_highlevel_bev = self.hightlevel_bev_kd_loss(student_bev, teacher_bev)
            loss_highlevel_bev_list.append(loss_highlevel_bev)

        else:
            if self.high_config.duplicate_highlevl_stu_feat:
                # img_feats_kd = torch.cat([img_feats_kd, img_feats_kd], dim=1)
                student_bev = student_bev.repeat((1, 2, 1, 1))
                loss_highlevel_bev = self.hightlevel_bev_kd_loss(
                    student_bev, teacher_bev
                )
                loss_highlevel_bev_list.append(loss_highlevel_bev)
            else:
                self.inter_keypoint_weight = 100.0
                self.inter_channel_weight = 10.0

                if self.inter_keypoint_weight > 0 or self.inter_channel_weight > 0:
                    student_bev, teacher_bev = self.get_inner_feat(
                        gt_bboxes_3d, student_bev, teacher_bev
                    )

                if self.inter_keypoint_weight > 0:
                    loss_inter_keypoint = self.get_inter_keypoint_loss(
                        student_bev, teacher_bev
                    )
                    loss_highlevel_bev_list.append(loss_inter_keypoint)

                if self.inter_channel_weight > 0:
                    loss_inter_channel = self.get_inter_channel_loss(
                        student_bev, teacher_bev
                    )
                    loss_highlevel_bev_list.append(loss_inter_channel)

        return loss_highlevel_bev_list

    def unify_feat_size(self, student_feat, teacher_feat):
        bs, s_c, s_h, s_w = student_feat.shape
        bs, t_c, t_h, t_w = teacher_feat.shape
        if s_w == t_w:
            return student_feat, teacher_feat
        else:
            teacher_feat = F.interpolate(teacher_feat, (s_h, s_w), mode="bilinear")
            return student_feat, teacher_feat

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img_inputs=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        low_level_distill_cfg = self.distill_config.low_level_distill

        (
            pts_feats,
            teacher_low_level_feats,
            teacher_desne_ms_bev_feats,
            tea_feats_dict,
        ) = self.get_teacher_kd(points, img_inputs, img_metas, gt_bboxes_3d)

        img_feats, depth, feats_dict = self.extract_feat(points, img_inputs, img_metas)

        assert self.with_pts_bbox
        depth_gt = img_inputs[6]
        losses = dict()

        if self.distill_config.is_depth_supervise:
            if self.img_view_transformer.loss_depth_weight > 0:
                loss_depth = self.get_depth_loss(depth_gt, depth)
                losses.update({"loss_depth": loss_depth})

        if pts_feats:
            pts_feats_kd = pts_feats[0]

        img_feats_kd = img_feats

        losses_pts = self.forward_pts_train(
            [img_feats], gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
        )
        if self.distill_config.get("voxel_feats_distill", None):
            loss_voxel_feats = self.voxel_feats_distill(
                gt_bboxes_3d,
                gt_labels_3d,
                tea_feats_dict[2].get("low_ms_voxel_feats"),
                feats_dict.get("stu_voxel_feats"),
                # self.distill_feats.get("stu_voxel_feats"),
            )

            losses.update({"loss_voxel_feats_distill": loss_voxel_feats})

        if self.is_low_distill:
            if low_level_distill_cfg.low_level_distill_type == "2d":
                loss_low_level_feats_distill = self.get_low_level_feats_distill(
                    feats_dict,
                    teacher_low_level_feats,
                    loss_type=self.low_config.loss_type,
                    gt_boxes_3d=gt_bboxes_3d,
                )

            losses.update(
                {
                    "loss_low_level_%s_feats_inner_kd"
                    % (
                        low_level_distill_cfg.low_level_distill_type
                    ): loss_low_level_feats_distill
                }
            )
        if self.is_ms_distill:
            loss_ms_kd = self.get_multi_scale_dense_distill(
                ms_stu_list=feats_dict["multi_scale_feats"],
                ms_tea_list=teacher_desne_ms_bev_feats,
            )
            losses.update({"loss_multi_scale_kd": loss_ms_kd})

        if self.is_high_distill:
            loss_highlevel_bev = self.get_high_level_bev_kd_loss(
                img_feats_kd, pts_feats_kd, gt_bboxes_3d
            )

            if len(loss_highlevel_bev) > 1:
                losses.update({"loss_high_inner_keypoint": loss_highlevel_bev[0]})
                losses.update({"loss_high_inner_channel": loss_highlevel_bev[1]})
            else:
                losses.update({"loss_hightlevel_bev": loss_highlevel_bev[0]})

        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        for var, name in [(img_inputs, "img_inputs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(img_inputs), len(img_metas)
                )
            )

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get("combine_type", "output")
        if combine_type == "output":
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type == "feature":
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]

        if img_feats.shape.__len__() != 5:
            img_feats = img_feats.view(-1, *img_feats.shape)

        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list

    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes

        img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
