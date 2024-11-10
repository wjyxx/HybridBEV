_base_ = [
    "../_base_/datasets/nus-3d.py",
    "../_base_/default_runtime.py",
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
downsample = 16

image_scale = 0.447
data_config = {
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    "image_scale": False,
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

depth_thres = {
    "CAM_FRONT": 1,
    "CAM_FRONT_LEFT": 0,
    "CAM_FRONT_RIGHT": 0,
    "CAM_BACK": 0.5,
    "CAM_BACK_LEFT": 0,
    "CAM_BACK_RIGHT": 0,
}
num_points = 5

# Model
grid_config = {
    "xbound": [-51.2, 51.2, 0.8],
    "ybound": [-51.2, 51.2, 0.8],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]
# voxel_size = [0.075, 0.075, 0.2]

numC_Trans = 64

model = dict(
    type="HybirdBEV",
    is_distill=True,
    img_backbone=dict(
        pretrained="torchvision://resnet50",
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style="pytorch",
    ),
    img_neck=dict(
        type="FPNForBEVDet",
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    img_view_transformer=dict(
        type="ViewTransformerLSSBEVDepth",  # ViewTransformerLSSBEVDepth, ViewTransformerLiftSplatShoot
        loss_depth_weight=100.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        extra_depth_net=dict(
            type="ResNetForBEVDet",
            numC_input=256,
            num_layer=[
                3,
            ],
            num_channels=[
                256,
            ],
            stride=[
                1,
            ],
        ),
    ),
    hybird_encoder=dict(
        type="HybirdEncoder",
        num_cams=6,
        num_convs=3,
        num_points=num_points,
        num_sweeps=1,
        kernel_size=(3, 3, 3),
        keep_sweep_dim=True,
        num_feature_levels=1,
        embed_dims=128,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_shape=[128, 128, num_points],
        fp16_enabled=True,
        encoding=True,
        conv_dict=dict(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d"),
            act_cfg=dict(type="ReLU"),
            inplace=True,
        ),
    ),
    pre_process=dict(type="SCNET", input_channels=64, num_layers=5),
    img_bev_encoder_backbone=dict(type="ResNetForBEVDet", numC_input=numC_Trans),
    img_bev_encoder_neck=dict(
        type="FPN_LSS",
        in_channels=(numC_Trans) * 8 + (numC_Trans) * 2,
        out_channels=256,
    ),
    pts_bbox_head=dict(
        type="CenterHead_task",
        task_specific=True,
        in_channels=sum([256, 0]),  # if 'dual_feats_fusion cat fusion' 512 else 256
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            # Scale-NMS
            nms_type=["rotate", "rotate", "rotate", "circle", "rotate", "rotate"],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0,
                [0.7, 0.7],
                [0.4, 0.55],
                1.1,
                [1.0, 1.0],
                [4.5, 9.0],
            ],
        )
    ),
    distill_config=dict(
        teacher_type="lidar",
        freeze_teacher=True,
        stu_load_tea_head=False,
        is_depth_supervise=True,
        voxel_feats_distill=dict(
            type="VoxelDistill",
            criterion="mse",
            loss_weight=100,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            class_names=class_names,
            task_num=6,
        ),
        dfd_distill=dict(
            low_level_distill_loss=dict(
                type="CriterionCWD_KD",
                norm_type="channel",
                divergence="kl",
                temperature=4,
                params_weight=10,
            ),
            loss_type="bev-v1",
        ),
        sd_distill=dict(
            is_multi_scale_distill=False,
            multi_scale_distill_loss=dict(
                type="MultiScaleFeatsDistill",
                lvls=2,
                compress="mean",
                lossMeanWeight=10,  # 1, 10, 100
                dist=True,
            ),
        ),
        high_level_distill=dict(
            is_high_level_distill=False,
            duplicate_highlevl_stu_feat=False,
            high_level_distill_loss=dict(
                type="SimpleMSE", param_weight=10, compress="mean"
            ),
        ),
        s_conv_modify=dict(in_channel=[16, 32, 64, 128]),
    ),
    teacher_config_file="path/to/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py",
    teacher_pretrained="path/to/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth",
)


dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

train_pipeline = [
    dict(
        type="LoadMultiViewImageFromFiles_BEVDet",
        is_train=True,
        data_config=data_config,
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointToMultiViewDepth", grid_config=grid_config),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["points", "img_inputs", "gt_bboxes_3d", "gt_labels_3d"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_BEVDet", data_config=data_config),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="PointToMultiViewDepth", grid_config=grid_config),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(type="Collect3D", keys=["points", "img_inputs"]),
        ],
    ),
]
eval_pipeline = [
    dict(type="LoadMultiViewImageFromFiles_BEVDet", data_config=data_config),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="PointToMultiViewDepth", grid_config=grid_config),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["img_inputs"]),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    dataset_scale=8,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            box_type_3d="LiDAR",
            img_info_prototype="bevdet",
        ),
    ),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
    ),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        img_info_prototype="bevdet",
    ),
)


evaluation = dict(interval=1, pipeline=test_pipeline)

# Optimizer
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=200, warmup_ratio=0.001, step=[16, 19]
)
runner = dict(type="EpochBasedRunner", max_epochs=20)
work_dir = "hybirdbev/base"
