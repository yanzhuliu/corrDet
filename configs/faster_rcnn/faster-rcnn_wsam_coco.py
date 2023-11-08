_base_ = [
    '../_base_/models/faster-rcnn_wsam.py',
    '../_base_/datasets/coco_detection_sam.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer
optim_wrapper = dict(
    type='SamOptimWrapper',
    optimizer=dict(type='WSAM', rho=0.05, base_optimizer='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'