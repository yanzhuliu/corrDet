# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.runner import TestLoop
from mmdet.registry import LOOPS

# by lyz
from mmengine.runner.amp import autocast
from typing import Dict, List, Sequence, Union
from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator
import torch

@LOOPS.register_module()
class TestLossLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self._max_epoch = runner.optim_wrapper.optimizer.x_num * runner.optim_wrapper.optimizer.y_num
        self._epoch = 0

    def run(self) -> None:
        self.runner.optim_wrapper = self.runner.build_optim_wrapper(self.runner.optim_wrapper)
        self.runner.call_hook('before_test')

        while self._epoch < self._max_epoch:
            self.run_epoch()
        self.runner.call_hook('after_test')


    def run_epoch(self) -> dict:
        """Iterate one epoch."""
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self.runner.optim_wrapper.update_params(loss=None)
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute total of whole epoch
        losses = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=losses)
        print('epoch: ' + str(self._epoch))
        self._epoch += 1
        return losses

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be dict of loss
        with autocast(enabled=self.fp16):
            losses = self.runner.model.test_loss_step(data_batch)
        self.evaluator.process(data_samples=[losses], data_batch=data_batch)
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=losses)