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

train_stylized_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
#    dict(type='AugMix', mixture_width=3, aug_severity=3, jsd_loss=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_clean_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
#    dict(type='AugMix', mixture_width=3, aug_severity=3, jsd_loss=False),  # by lyz, only augmix clean images
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

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

train_dataloader = dict(
    batch_size=10,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_clean_pipeline,
                    backend_args=backend_args),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_clean_pipeline,
                    backend_args=backend_args),
    #            dict(
    #                type=dataset_type,
    #                data_root=data_root,
    #                ann_file='VOC2007/ImageSets/Main/trainval.txt',
    #                data_prefix=dict(sub_data_root='VOC2007_stylize/'),
    #                filter_cfg=dict(
    #                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
    #                pipeline=train_stylized_pipeline,
    #                backend_args=backend_args),
    #            dict(
    #                type=dataset_type,
    #                data_root=data_root,
    #                ann_file='VOC2012/ImageSets/Main/trainval.txt',
    #                data_prefix=dict(sub_data_root='VOC2012_stylize/'),
    #                filter_cfg=dict(
    #                    filter_empty_gt=True, min_size=32, bbox_min_size=32),
    #                pipeline=train_stylized_pipeline,
    #                backend_args=backend_args)
            ])))

val_dataloader = dict(
    batch_size=1,
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
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(
            filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args),
    )#)

test_dataloader = val_dataloader

#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2007/ImageSets/Main/trainval.txt',
#                     data_prefix=dict(sub_data_root='VOC2007_stylize/'),
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=32, bbox_min_size=32),
#                     pipeline=train_pipeline,
#                     backend_args=backend_args),
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2012/ImageSets/Main/trainval.txt',
#                     data_prefix=dict(sub_data_root='VOC2012_stylize/'),
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=32, bbox_min_size=32),
#                     pipeline=train_pipeline,
#                     backend_args=backend_args)
    # dataset=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file='VOC2007-test/ImageSets/Main/test.txt',
    #     data_prefix=dict(sub_data_root='voc07test-zoom_blur-1/'),  # 'VOC2007-test'),#
    #     test_mode=True,
    #     pipeline=test_pipeline,
    #     backend_args=backend_args))

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
