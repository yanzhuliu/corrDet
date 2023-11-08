_base_ = [
    '../_base_/models/faster-rcnn_wsam.py',
    '../_base_/datasets/voc0712_sam.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 4
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# by lyz
optim_wrapper = dict(
   type='SamOptimWrapper',
   optimizer=dict(type='WSAM', rho=0.1, adaptive=True, base_optimizer='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
load_from = '../checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth'