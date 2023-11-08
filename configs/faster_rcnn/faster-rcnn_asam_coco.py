_base_ = [
    '../_base_/models/faster-rcnn_sam.py',
    '../_base_/datasets/coco_detection_sam.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(
    batch_size=32,
    num_workers=3
)

# optimizer
optim_wrapper = dict(
    type='SamOptimWrapper',
    optimizer=dict(type='SAM', rho=0.5, adaptive=True, base_optimizer='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'