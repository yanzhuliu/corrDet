from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
import torch
from typing import Dict, Optional

@OPTIM_WRAPPERS.register_module()
class LossOptimWrapper(OptimWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_params(self,
                      loss: torch.Tensor,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None):

        self.optimizer.step()