dataset_type = 'OccScanNetDataset'
dataset_root = 'data/occscannet'
pretrained_depth_model = 'Metric3Dv2-Giant'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

occ_names = [
    'others', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furniture', 'objects'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [0.0, 0.0, 0.0, 4.8, 4.8, 2.88]
voxel_size = [0.08, 0.08, 0.08]

# arch config
embed_dims = 256
num_layers = 6
num_query = 960
num_levels = 4
num_points = 16
num_refines = [1, 2, 4, 8, 16, 16]

num_frames = 1
offset = 0
ignore_idx = 0
empty_idx = 12   # 0 ignore, 1~11 objects, 12 empty
num_anchor_init = num_query
resize_lim = [1.0, 1.0]

pretrained = 'pretrain/upernet_internimage_xl_640_160k_ade20k.pth'
img_backbone = dict(
    type='InternImage',
    core_op='DCNv3',
    channels=192,
    depths=[5, 5, 24, 5],
    groups=[12, 24, 48, 96],
    mlp_ratio=4.,
    drop_path_rate=0.4,
    norm_layer='LN',
    layer_scale=1.0,
    offset_scale=2.0,
    post_norm=True,
    with_cp=False,
    out_indices=(0, 1, 2, 3),
    init_cfg=dict(type='Pretrained', checkpoint=pretrained)
)
img_neck = dict(
    type='FPN',
    in_channels=[192, 384, 768, 1536],
    out_channels=embed_dims,
    num_outs=num_levels)

model = dict(
    type='DiScene',
    use_grid_mask=False,
    data_aug=None,
    stop_prev_grad=0,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='DiSceneHead_Teacher',
        num_classes=len(occ_names),
        in_channels=embed_dims,
        num_query=num_query,
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        transformer=dict(
            type='DiSceneTransformer_MD',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=len(occ_names),
            num_refines=num_refines,
            scales=[0.5],
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.04, loss_weight=0.5),
        pretrained_depth_model=pretrained_depth_model),
    train_cfg=dict(
        pts=dict(
            cls_freq=[4393702080, 700497, 44393812, 39153315, 3367920, 21860824, 10667655, 13832443, 23358785, 263684, 30415451, 9338079],
            cls_freq_offset=0.05,
            empty_dist_thr=0.04,
            empty_weights=5,
            rare_classes=[1, 4, 6, 9, 11],  # 'others', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furniture', 'objects'
            rare_weights=10,
            )
        ),
    test_cfg=dict(
        pts=dict(
            ctr_dist_thr=0.6,
            score_thr=0.5,
            padding=True
        )
    )
)

data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_path=dataset_root,
        num_frames=num_frames,
        offset=offset,
        empty_idx=empty_idx,
        phase='train',
        final_dim=[480, 640], 
        resize_lim=resize_lim,
        num_pts=num_anchor_init,
        pretrained_depth_model=pretrained_depth_model,
        data_tg='base',
        ),
    val=dict(
        type=dataset_type,
        data_path=dataset_root,
        num_frames=num_frames,
        offset=offset,
        empty_idx=empty_idx,
        phase='test',
        final_dim=[480, 640], 
        resize_lim=resize_lim,
        num_pts=num_anchor_init,
        pretrained_depth_model=pretrained_depth_model,
        data_tg='base',
        ),
    test=dict(
        type=dataset_type,
        data_path=dataset_root,
        num_frames=num_frames,
        offset=offset,
        empty_idx=empty_idx,
        phase='test',
        final_dim=[480, 640], 
        resize_lim=resize_lim,
        num_pts=num_anchor_init,
        pretrained_depth_model=pretrained_depth_model,
        data_tg='base',
        )
)

optimizer = dict(
    type='AdamW',
    constructor='CustomOptimizerConstructor',
    lr=2e-4,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='OptimizerHook',
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 10
batch_size = 8

# load pretrained weights
# load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
# revise_keys = [('backbone', 'img_backbone')]
load_from = None

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=5, max_keep_ckpts=10)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=50, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=50, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=10)

# other flags
debug = False
