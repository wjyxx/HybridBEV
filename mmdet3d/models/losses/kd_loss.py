import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.builder import LOSSES
from mmdet3d.core import (
    draw_heatmap_gaussian,
    gaussian_radius,
    draw_heatmap_gaussian_3D,
)
from mmdet.core import multi_apply

from mmcv.runner import force_fp32, BaseModule


def unify_feat_size(student_feat, teacher_feat):
    """make the feats shape of teacher equals to the student, make the alignment"""
    bs, s_c, s_h, s_w = student_feat.shape
    bs, t_c, t_h, t_w = teacher_feat.shape
    if s_w == t_w:
        return student_feat, teacher_feat
    else:
        teacher_feat = F.interpolate(teacher_feat, (s_h, s_w), mode="bilinear")
        return student_feat, teacher_feat


@LOSSES.register_module()
class ChannelWiseDivergenceLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str):
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.

    """

    def __init__(
        self,
        student_channels,
        teacher_channels,
        name,
        tau=1.0,
        weight=1.0,
    ):
        super(ChannelWiseDivergenceLoss, self).__init__()
        self.tau = tau
        self.loss_weight = weight

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

    def forward(self, preds_S, preds_T):
        """Forward function."""
        assert (
            preds_S.shape[-2:] == preds_T.shape[-2:]
        ), "the output dim of teacher and student differ"
        N, C, W, H = preds_S.shape

        if self.align is not None:
            preds_S = self.align(preds_S)

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        softmax_pred_S = F.softmax(preds_S.view(-1, W * H) / self.tau, dim=1)

        # todo calculate the sofrmax_pred_s to the distill

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(
            -softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / self.tau)
        ) * (self.tau**2)
        return self.loss_weight * loss / (C * N)


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


@LOSSES.register_module()
class CriterionCWD_KD(nn.Module):

    def __init__(
        self, norm_type="none", divergence="mse", temperature=1.0, params_weight=1.0
    ):

        super(CriterionCWD_KD, self).__init__()

        # define normalize function
        if norm_type == "channel":
            self.normalize = ChannelNorm()
        elif norm_type == "spatial":
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == "channel_mean":
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == "mse":
            self.criterion = nn.MSELoss(reduction="sum")
        elif divergence == "kl":
            self.criterion = nn.KLDivLoss(reduction="sum")
            self.temperature = temperature
        self.divergence = divergence
        self.params_weight = params_weight

    def forward(self, preds_S, preds_T):

        s_h = preds_S.size(2)
        t_h = preds_T.size(2)

        if s_h != t_h:
            _, preds_T = unify_feat_size(preds_S, preds_T)

        n, c, h, w = preds_S.shape
        # import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == "kl":
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == "channel" or self.norm_type == "channel_mean":
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2) * self.params_weight


@LOSSES.register_module()
class MSEWithDist(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """

    def __init__(
        self,
        student_channels,
        teacher_channels,
        name=None,
        alpha_mgd=0.00002,
        lambda_mgd=0.65,
    ):
        super(MSEWithDist, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

    def forward(self, preds_S, preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction="sum")
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


@LOSSES.register_module()
class SimpleBCE(nn.Module):
    def __init__(self, reduction=None, param_weight=1):

        super(SimpleBCE, self).__init__()
        self.reduction = reduction
        self.param_weight = param_weight
        self.criterion = nn.BCELoss()

    def forward(self, preds_S, preds_T):

        loss = 0
        loss += self.criterion(preds_S, preds_T)

        return loss * self.param_weight


@LOSSES.register_module()
class SimpleMSE(nn.Module):
    def __init__(self, size_average=None, reduce=None, compress="mean", param_weight=1):

        super(SimpleMSE, self).__init__()
        self.param_weight = param_weight
        self.criterion = nn.MSELoss(size_average=size_average, reduction=compress)

    def forward(self, preds_S, preds_T):

        # print(f"==={preds_S.shape}==={preds_T.shape}====")
        # loss = 0
        # loss = (self.criterion(preds_S, preds_T)).mean() * 32
        loss = F.mse_loss(preds_S, preds_T)
        loss = torch.mean(loss)

        return loss * self.param_weight


@LOSSES.register_module()
class FeatsKDSmoothL1(nn.Module):
    def __init__(
        self, size_average=None, reduce=None, reduction="mean", param_weight=1
    ):

        super(FeatsKDSmoothL1, self).__init__()
        self.param_weight = param_weight
        self.criterion = nn.SmoothL1Loss(reduce=reduction)

    def forward(self, preds_S, preds_T):

        # loss = 0
        loss = self.criterion(preds_S, preds_T)

        return loss * self.param_weight


@LOSSES.register_module()
class MultiScaleFeatsDistill(nn.Module):
    def __init__(self, lvls=4, compress="mean", lossMeanWeight=1, dist=False):
        super(MultiScaleFeatsDistill, self).__init__()
        self.num_lvls = lvls
        self.compress = compress
        self.lossMeanWeight = lossMeanWeight
        self.dist = dist

        # NOTE Manual configuration is recommended to prevent cuda core issues.
        if dist:
            self.criterion_loss_0 = MSEWithDist(128, 128, None, 2e-5)
            self.criterion_loss_1 = MSEWithDist(128 * 2, 256, None, 2e-5)
            # self.criterion_loss_0 = FeatsNormDistill(
            #     student_channel=128, teacher_channel=128
            # )
            # self.criterion_loss_1 = FeatsNormDistill(
            #     student_channel=128 * 2, teacher_channel=256
            # )
        else:
            self.criterion_loss = nn.MSELoss()

    def forward(self, ms_stu, ms_tea):
        loss = 0
        if self.dist:
            for idx in range(self.num_lvls):
                loss += self.__getattr__(f"criterion_loss_{idx}")(
                    ms_stu[idx], ms_tea[idx]
                )
        else:
            for idx in range(self.num_lvls):
                loss += self.criterion_loss(ms_stu[idx], ms_tea[idx])

        if self.compress == "mean":
            loss = (loss / self.num_lvls) * self.lossMeanWeight
        else:
            loss = loss * self.lossMeanWeight

        return loss


@LOSSES.register_module()
class MoCoC_Head_KD(BaseModule):
    def __init__(
        self,
        img_channels,
        pts_channels,
        mid_channels=128,
        img_proj_num=1,
        pts_proj_num=1,
        T=0.07,
        loss_cl=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ):
        super().__init__()
        img_projs = []
        pts_projs = []
        img_input_channels = img_channels
        pts_input_channels = pts_channels
        for ii in range(img_proj_num):
            img_proj = nn.Sequential(
                nn.Linear(img_input_channels, mid_channels),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            )
            img_input_channels = mid_channels
            img_projs.append(img_proj)
        for ii in range(pts_proj_num):
            pts_proj = nn.Sequential(
                nn.Linear(pts_input_channels, mid_channels),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            )
            pts_input_channels = mid_channels
            pts_projs.append(pts_proj)
        self.img_projs = nn.ModuleList(img_projs)
        self.pts_projs = nn.ModuleList(pts_projs)
        # 2 layer mlp encoder
        self.encoder_img = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
        )
        self.encoder_pts = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
        )
        self.mid_channels = mid_channels
        self.T = T
        self.loss_cl = nn.CrossEntropyLoss()

    # @force_fp32(apply_to=('logits', 'labels'))
    # def loss(self, logits, labels):
    #     loss_cl = self.loss_cl(logits, labels)
    #     return loss_cl

    def forward(self, img_feats, pts_feats):
        # print(f":========img size :===={img_feats.shape}===========pts size:=={pts_feats.shape}============")
        """计算faltten后的特征概率"""
        for pts_proj in self.pts_projs:
            pts_feats = pts_proj(pts_feats)
        for img_proj in self.img_projs:
            img_feats = img_proj(img_feats)

        pts_feats = self.encoder_pts(pts_feats)
        pts_feats = F.normalize(pts_feats, dim=1)

        img_feats = self.encoder_img(img_feats)
        img_feats = F.normalize(img_feats, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        logits = torch.einsum("nc,ck->nk", [pts_feats, img_feats.T])
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.arange(logits.shape[0]).cuda(logits.device)
        loss_cl = self.loss_cl(logits, labels)
        return loss_cl


@LOSSES.register_module()
class InfoMax_Loss(nn.Module):
    """Info Max Theory Loss for Distillation
    Args:
        loss_weight (int): loss weight for traing
    """

    def __init__(self, loss_weight=1.0):
        super(InfoMax_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward_single(self, x1, x2):
        x1 = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True) + 1e-10)
        x2 = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True) + 1e-10)
        bs = x1.size(0)
        s = torch.matmul(x1, x2.permute(1, 0))
        mask_joint = torch.eye(bs).cuda()
        mask_marginal = 1 - mask_joint

        Ej = (s * mask_joint).mean()
        Em = torch.exp(s * mask_marginal).mean()
        # decoupled comtrastive learning?!!!!
        # infomax_loss = - (Ej - torch.log(Em)) * self.alpha
        infomax_loss = -(Ej - torch.log(Em))  # / Em
        return infomax_loss * self.loss_weight

    def forward(self, x_stu, x_tea):
        """forward, for 2d bev
        Args:
            x1 (torch.tensor)
            x2 (torch.tensor)
        """
        assert len(x_stu.size()) >= 4
        assert len(x_tea.size()) >= 4
        B, C, H, W = x_stu.size()
        # x_stu=

        return self.forward_single(x_stu, x_tea)


@LOSSES.register_module()
class FeatsNormDistill(nn.Module):
    """refs from PKD

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(
        self, loss_weight=1.0, resize_stu=True, student_channel=64, teacher_channel=256
    ):
        super(FeatsNormDistill, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

        if student_channel != teacher_channel:
            self.align = nn.Conv2d(
                student_channel, teacher_channel, padding=0, stride=1, kernel_size=1
            )

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S,), (preds_T,)

        loss = 0.0

        for pred_S, pred_T in zip(preds_S, preds_T):

            if pred_S.size(1) != pred_T.size(1):
                pred_S = self.align(pred_S)

            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode="bilinear")
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode="bilinear")

            assert pred_S.shape == pred_T.shape

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += F.mse_loss(norm_S, norm_T) / 2

        return loss * self.loss_weight


class Result_Level_Distill(nn.Module):
    def __init__(self, pc_range, voxel_size, out_size_scale):
        super().__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.out_size_scale = out_size_scale

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-3, max=1 - 1e-3)
        return y

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        if (
            min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
        ):  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def calculate_box_mask_gaussian(
        self, preds_shape, target, pc_range, voxel_size, out_size_scale
    ):
        B = preds_shape[0]
        C = preds_shape[1]
        H = preds_shape[2]
        W = preds_shape[3]
        gt_mask = np.zeros((B, H, W), dtype=np.float32)  # C * H * W

        for i in range(B):
            for j in range(len(target[i])):
                if target[i][j].sum() == 0:
                    break

                w, h = (
                    target[i][j][3] / (voxel_size[0] * out_size_scale),
                    target[i][j][4] / (voxel_size[1] * out_size_scale),
                )
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))

                center_heatmap = [
                    int(
                        (target[i][j][0] - pc_range[0])
                        / (voxel_size[0] * out_size_scale)
                    ),
                    int(
                        (target[i][j][1] - pc_range[1])
                        / (voxel_size[1] * out_size_scale)
                    ),
                ]
                self.draw_umich_gaussian(gt_mask[i], center_heatmap, radius)

        gt_mask_torch = torch.from_numpy(gt_mask).cuda()
        return gt_mask_torch

    def get_loss(self, stu_feats_results, tea_feats_result, gt_boxes):
        cls_lidar = []
        reg_lidar = []
        cls_fuse = []
        reg_fuse = []
        criterion = nn.L1Loss(reduce=False)
        for task_id, task_out in enumerate(stu_feats_results):
            cls_lidar.append(task_out["hm"])
            cls_fuse.append(self._sigmoid(tea_feats_result[task_id]["hm"] / 2))
            reg_lidar.append(
                torch.cat(
                    [
                        task_out["reg"],
                        task_out["height"],
                        task_out["dim"],
                        task_out["rot"],
                        task_out["vel"],
                        task_out["iou"],
                    ],
                    dim=1,
                )
            )
            reg_fuse.append(
                torch.cat(
                    [
                        tea_feats_result[task_id]["reg"],
                        tea_feats_result[task_id]["height"],
                        tea_feats_result[task_id]["dim"],
                        tea_feats_result[task_id]["rot"],
                        tea_feats_result[task_id]["vel"],
                        tea_feats_result[task_id]["iou"],
                    ],
                    dim=1,
                )
            )
        cls_lidar = torch.cat(cls_lidar, dim=1)
        reg_lidar = torch.cat(reg_lidar, dim=1)
        cls_fuse = torch.cat(cls_fuse, dim=1)
        reg_fuse = torch.cat(reg_fuse, dim=1)
        cls_lidar_max, _ = torch.max(cls_lidar, dim=1)
        cls_fuse_max, _ = torch.max(cls_fuse, dim=1)
        gaussian_mask = self.calculate_box_mask_gaussian(
            reg_lidar.shape,
            gt_boxes.cpu().detach().numpy(),
            self.pc_range,
            self.voxel_size,
            self.out_size_scale,
        )
        diff_reg = criterion(reg_lidar, reg_fuse)
        diff_cls = criterion(cls_lidar_max, cls_fuse_max)
        diff_reg = torch.mean(diff_reg, dim=1)
        diff_reg = diff_reg * gaussian_mask
        diff_cls = diff_cls * gaussian_mask
        weight = gaussian_mask.sum()
        # weight = reduce_mean(weight)
        weight = weight.mean()
        loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
        loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)
        return loss_cls_distill, loss_reg_distill

    def forward(self, stu_feats_results, tea_feats_result, gt_boxes):
        loss_cls_distill, loss_reg_distill = self.get_loss(
            stu_feats_results=stu_feats_results,
            tea_feats_result=tea_feats_result,
            gt_boxes=gt_boxes,
        )
        return loss_cls_distill, loss_reg_distill


@force_fp32(apply_to=("student_bev_feat", "teacher_bev_feat", "sampling_points"))
def get_bev_fit_loss(
    self,
    student_bev_feat,
    teacher_bev_feat,
    gt_bboxes_list=None,
    fg_pred_map=None,
    sampling_points=None,
):
    if sampling_points is not None:
        sampling_points = sampling_points.permute(1, 0, 2, 3)
        select_student_feat = F.grid_sample(student_bev_feat, sampling_points)
        select_teacher_feat = F.grid_sample(teacher_bev_feat, sampling_points)
        fit_loss = F.mse_loss(select_student_feat, select_teacher_feat)
        fit_loss = torch.mean(fit_loss)
        return dict(bev_fit_loss=fit_loss), None

    fg_map = None
    if gt_bboxes_list is not None and fg_pred_map is not None:
        raise Exception("distill fg weight should be specified!")
    grid_size = torch.tensor(self.grid_size)
    pc_range = torch.tensor([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    voxel_size = torch.tensor(self.voxel_size)
    feature_map_size = grid_size[:2] // 8
    if gt_bboxes_list is not None:
        device = student_bev_feat.device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]
        fg_map = student_bev_feat.new_zeros(
            (len(gt_bboxes_list), feature_map_size[1], feature_map_size[0])
        )
        for idx in range(len(gt_bboxes_list)):
            num_objs = gt_bboxes_list[idx].shape[0]
            for k in range(num_objs):
                width = gt_bboxes_list[idx][k][3]
                length = gt_bboxes_list[idx][k][4]
                width = width / voxel_size[0] / 8
                length = length / voxel_size[1] / 8

                if width > 0 and length > 0:
                    radius = gaussian_radius((length, width), min_overlap=0.1)
                    radius = max(1, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = (
                        gt_bboxes_list[idx][k][0],
                        gt_bboxes_list[idx][k][1],
                        gt_bboxes_list[idx][k][2],
                    )

                    coor_x = (x - pc_range[0]) / voxel_size[0] / 8
                    coor_y = (y - pc_range[1]) / voxel_size[1] / 8

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue

                    draw_heatmap_gaussian(fg_map[idx], center_int, radius)
    if fg_pred_map is not None:
        fg_map = torch.max(fg_pred_map.sigmoid(), dim=1)[0]

    if fg_map is None:
        fg_map = student_bev_feat.new_ones(
            (student_bev_feat.shape[0], feature_map_size[1], feature_map_size[0])
        )
    fit_loss = F.mse_loss(student_bev_feat, teacher_bev_feat, reduction="none")
    fit_loss = torch.sum(fit_loss * fg_map) / torch.sum(fg_map)
    return dict(bev_fit_loss=fit_loss), fg_map


@LOSSES.register_module()
class VoxelDistill(nn.Module):
    def __init__(
        self,
        criterion=None,
        reduction="none",
        loss_weight=10,
        train_cfg=None,
        class_names=None,
        task_num=None,
    ):
        super().__init__()

        self.loss_weight = loss_weight

        if criterion == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError

        self.train_cfg = train_cfg
        self.class_names = class_names
        self.task_num = task_num

        self.align_ch = nn.Conv3d(64, 128, stride=1, kernel_size=1, padding=0)

    def align(self, voxels, shape):
        if voxels.size(1) == 64:
            voxels = self.align_ch(voxels)
        voxels = F.interpolate(voxels, shape, mode="trilinear")

        return voxels

    @force_fp32(apply_to=("teacher_voxel_feats", "student_voxel_feats"))
    def forward(
        self, gt_bboxes_3d, gt_labels_3d, teacher_voxel_feats, student_voxel_feats
    ):

        heatmaps3D = multi_apply(
            self.get_3d_gaussian_mask_single, gt_bboxes_3d, gt_labels_3d
        )  # [LiDARInstance3DBoxes...32e-02]]))], [tensor([8, 8, 0, 8, ...='cuda:0')]
        # Transpose heatmaps
        heatmaps3D = list(map(list, zip(*heatmaps3D)))
        heatmaps3D = [
            torch.stack(hms_) for hms_ in heatmaps3D
        ]  # [torch.Size([bs, 256, 256, 8])]
        heatmaps3D = torch.stack(heatmaps3D).permute(0, 1, 4, 2, 3)

        assert teacher_voxel_feats.size(4) == heatmaps3D.size(3)
        student_voxel_feats = self.align(
            student_voxel_feats, teacher_voxel_feats.shape[2:]
        )
        assert student_voxel_feats.size(4) == heatmaps3D.size(3)

        # loss_voxel_distill = self.criterion(
        #     student_voxel_feats * heatmaps3D, teacher_voxel_feats * heatmaps3D
        # )
        loss_voxel_distill = (
            self.criterion(student_voxel_feats, teacher_voxel_feats) * heatmaps3D
        )
        # loss_voxel_distill = loss_voxel_distill * heatmaps3D
        average_weight = heatmaps3D.sum()
        loss_voxel_distill = loss_voxel_distill / (average_weight + 1e-4)

        return loss_voxel_distill * self.loss_weight

    def get_3d_gaussian_mask_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:

        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(
            self.train_cfg["point_cloud_range"]
        )  # tensor([-51.2, -51.2,  -5.0,  51.2,  51.2,   3.0])
        voxel_size = torch.tensor(
            self.train_cfg["voxel_size"]
        )  # tensor([0.1, 0.1, 0.2])

        feature_map_size = grid_size[:2] // self.train_cfg["out_size_factor"]

        voxel_resolution = {
            "x": [-51.2, 51.2, 0.8],
            "y": [-51.2, 51.2, 0.8],
            "z": [-5, 3, 4],
        }
        range_x = [
            0,
            int(
                (voxel_resolution["x"][1] - voxel_resolution["x"][0])
                / voxel_resolution["x"][2]
            )
            - 1,
        ]
        range_y = [
            0,
            int(
                (voxel_resolution["y"][1] - voxel_resolution["y"][0])
                / voxel_resolution["y"][2]
            )
            - 1,
        ]
        range_z = [
            0,
            int(
                (voxel_resolution["z"][1] - voxel_resolution["z"][0])
                / voxel_resolution["z"][2]
            )
            - 1,
        ]
        heatmap_occupancy = torch.zeros(
            (range_x[1] + 1, range_y[1] + 1, range_z[1] + 1)
        ).to(device)

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:  # class_name: ['motorcycle', 'bicycle']
            task_masks.append(
                [
                    torch.where(gt_labels_3d == class_name.index(i) + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)

        for idx in range(self.task_num):
            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                height = task_boxes[idx][k][5]
                voxel_resolution = {
                    "x": [-51.2, 51.2, 0.4],
                    "y": [-51.2, 51.2, 0.4],
                    "z": [-5, 3, 1],
                }

                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]
                height = height / voxel_size[2] / self.train_cfg["out_size_factor"]

                if width > 0 and length > 0:

                    radius = gaussian_radius(
                        (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius_x = radius
                    radius_y = radius
                    radius_z1 = gaussian_radius(
                        (length, height), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius_z2 = gaussian_radius(
                        (width, height), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius_z = max(radius_z1, radius_z2)
                    radius_x = int(radius_x / voxel_resolution["x"][2])
                    radius_y = int(radius_y / voxel_resolution["y"][2])
                    radius_z = int(radius_z / voxel_resolution["z"][2])
                    radius_x = max(self.train_cfg["min_radius"], radius_x)
                    radius_y = max(self.train_cfg["min_radius"], radius_y)
                    radius_z = max(self.train_cfg["min_radius"], radius_z)

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = (
                        task_boxes[idx][k][0],
                        task_boxes[idx][k][1],
                        task_boxes[idx][k][2],
                    )

                    coor_x = (x - voxel_resolution["x"][0]) / voxel_resolution["x"][2]
                    coor_y = (y - voxel_resolution["y"][0]) / voxel_resolution["y"][2]
                    coor_z = (z - voxel_resolution["z"][0]) / voxel_resolution["z"][2]

                    center = torch.tensor(
                        [coor_x, coor_y, coor_z], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)
                    radius = (radius_x, radius_y, radius_z)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                        and range_z[0] <= center_int[2] <= range_z[1]
                    ):
                        continue

                    draw_heatmap_gaussian_3D(heatmap_occupancy, center_int, radius)

                    # x, y = center_int[0], center_int[1]

                    # assert (y * feature_map_size[0] + x <
                    #         feature_map_size[0] * feature_map_size[1])

        return (heatmap_occupancy,)
