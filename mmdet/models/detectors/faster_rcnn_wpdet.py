# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from typing import Dict, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.utils import is_list_of

@MODELS.register_module()
class WpDetFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

    def disable_running_stats(self):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0
        self.apply(_disable)

    def enable_running_stats(self):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum
        self.apply(_enable)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            data['inputs'].requires_grad = True
            losses = self._run_forward(data, mode='loss')

        parsed_losses, log_vars = self.parse_losses(losses)
        data_adv = optim_wrapper.image_perturb(data, parsed_losses)
        data['inputs'].requires_grad = False
        optim_wrapper.update_params(parsed_losses)  # update for clean X

        with optim_wrapper.optim_context(self):
            losses = self._run_forward(data_adv, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)  # update for adv X

        # self.enable_running_stats()
        # with optim_wrapper.optim_context(self):
        #     data = self.data_preprocessor(data, True)
        #     losses = self._run_forward(data, mode='loss')
        # parsed_losses, log_vars = self.parse_losses(losses)
        # optim_wrapper.param_perturb(parsed_losses)
        #
        # self.disable_running_stats()
        # with optim_wrapper.optim_context(self):
        #     losses = self._run_forward(data_adv, mode='loss')
        # parsed_losses, log_vars = self.parse_losses(losses)
        # optim_wrapper.update_params(parsed_losses)

        #     for loss_name, loss_value in losses.items():
        #         if loss_name == "loss_rpn_cls":
        #             if isinstance(loss_value, torch.Tensor):
        #                 loss = loss_value.mean()
        #             elif is_list_of(loss_value, torch.Tensor):
        #                 loss = sum(_loss.mean() for _loss in loss_value)
        #             optim_wrapper.param_perturb(loss)

        return log_vars