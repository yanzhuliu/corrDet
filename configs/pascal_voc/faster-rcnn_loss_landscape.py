_base_ = [
    '../_base_/models/faster-rcnn_loss_landscape.py',
    '../_base_/datasets/voc0712_loss_landscape.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 4
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLossLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=500))

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

custom_hooks = [dict(type='PlotLossHook')]#, dict(type='EmptyCacheHook', after_iter=True)]
# optimizer
optim_wrapper = dict(
    type='LossOptimWrapper',
    optimizer=dict(type='LandScape', x_min=-0.5, x_max=0.5, x_num=31, y_min=-0.5, y_max=0.5, y_num=31))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
