_base_ = [
    # '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
fixed_ptsnum_per_gt_line = 30 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 30
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

# input_modality = dict(
#     use_lidar=False,
#     use_camera=True,
#     use_radar=False,
#     use_map=False,
#     use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 8
bev_w_ = 8
queue_length = 1 # each sequence contains `queue_length` frames.

# model = dict(
#     type='MapFusionTR',
#     vector_backbone=dict(
#         type='VectornetExtractor'),
model = dict(
    type='MapFusionTR',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    vector_backbone=dict(
        type='VectornetExtractor',
        static_input_dims = 6,
        node_num = 20,
        vector_num = 4,
        embedding_dims = 128,
        ff_dims = 256,
        map_num =2,
        attention_net=dict(
            type="StaticAttentionNet",
            gnn_encoder_layer_num=1,
            map_gnn_dim=256,
        )),
    pts_bbox_head=dict(
        type='MapFusionTRHead2',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=4,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapFusionTRPerceptionTransformer2',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapFusionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                # transformerlayers=dict(
                #     type='TransformerDecoderLayer',
                #     embed_dims=_dim_,
                #     nhead=8,
                #     dropout=0.1,
                #     dim_feedforward=2048,
                #     normalize_before=True)
                transformerlayers=dict(
                    type='MapFusionTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
                )
        ),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            # post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=10,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'), #这个cost不起作用，因为是作用到bbox的loss
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),#这个cost不起作用，因为是作用到bbox的loss
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))


dataset_type = 'BevFusionMapDataset'
data_root = 'data/mapfusion/maps'
file_client_args = dict(backend='disk')

load_features=['road_border']
train_pipeline = [
    dict(type='LoadMapFile', line_num = 4, node_num=30, load_features=load_features)
]


test_pipeline = [
    dict(type='LoadMapFile', line_num = 4, node_num=30, load_features=load_features)
] 
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        load_features=load_features,
        pipeline=train_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        queue_length=queue_length
        ),
    val=dict(type=dataset_type,
             root_dir=data_root,
              pipeline=test_pipeline, 
              load_features=load_features,
              bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
             test_mode = True),
    test=dict(type=dataset_type,
              root_dir=data_root,
              pipeline=test_pipeline, 
              load_features=load_features,
              bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
              test_mode = True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 200
# total_epochs = 50

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=1)
