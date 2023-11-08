from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
import torch
from typing import Dict, List, Optional

@OPTIM_WRAPPERS.register_module()
class ResOptimWrapper(OptimWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def param_perturb(self, loss: torch.Tensor,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)

        self.first_step(**step_kwargs)
        self.zero_grad(**zero_kwargs)

    def update_params(self,
                      loss: torch.Tensor,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None) -> None:

        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)

        if self.should_update():
            self.second_step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def first_step(self, **kwargs) -> None:
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.first_step(**kwargs)

    def second_step(self, **kwargs) -> None:
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.second_step(**kwargs)

