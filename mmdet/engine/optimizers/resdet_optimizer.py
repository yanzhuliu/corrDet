import copy

import torch
from mmdet.registry import OPTIMIZERS
import numpy as np

@OPTIMIZERS.register_module()
class ResDet(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, gamma=0.1, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.gamma = gamma
        defaults = dict(rho=rho, gamma=gamma, adaptive=adaptive, **kwargs)
        super(ResDet, self).__init__(params, defaults)

        if base_optimizer == 'SGD':
            self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        elif base_optimizer == 'Adam':
            self.base_optimizer = torch.optim.Adam(self.param_groups, **kwargs)
        elif base_optimizer == 'AdamW':
            self.base_optimizer = torch.optim.AdamW(self.param_groups, **kwargs)

        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            layer_idx = 0
            for p in group["params"]:
                layer_idx += 1
                if p.grad is None or p.dim() == 1 or layer_idx > 159: continue

                scale = group["rho"] / (((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2) + 1e-12)
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            layer_idx = 0
            for p in group["params"]:
                layer_idx += 1
                if p.grad is None or p.dim() == 1 or layer_idx > 159: continue

                if "old_p" not in self.state[p]:
                    print("no old_p")
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                self.state[p].pop("old_p")

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism

        norm_stack = []
        for group in self.param_groups:
            layer_idx = 0
            for p in group["params"]:
                layer_idx += 1
                if p.grad is None or layer_idx > 159:
                    continue
                norm_stack.append(
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                )

        norm = torch.norm(torch.stack(norm_stack), p=2)

        return norm
