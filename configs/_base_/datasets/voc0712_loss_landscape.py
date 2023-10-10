# dataset settings
dataset_type = 'VOCDataset'
data_root = '/mnt/data/causal_rodc/data/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically Infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/segmentation/VOCdevkit/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#         'data/': 's3://openmmlab/datasets/segmentation/'
#     }))
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
  #  dataset=dict(
  #      type='RepeatDataset',
  #      times=3,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(
            filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args),
    )#)

test_dataloader = val_dataloader
train_dataloader = val_dataloader

val_evaluator = dict(type='CollectLossMetric')  #by lyz
test_evaluator = val_evaluator
