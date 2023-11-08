import copy

import torch
from mmdet.registry import OPTIMIZERS
import numpy as np

@OPTIMIZERS.register_module()
class WpDet(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, gamma=8, eps=8, adaptive=False, **kwargs):
        assert eps >= 0.0, f"Invalid rho, should be non-negative: {eps}"
        self.gamma = gamma
        self.eps = eps
        defaults = dict(adaptive=adaptive, **kwargs)
        super(WpDet, self).__init__(params, defaults)

        if base_optimizer == 'SGD':
            self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        elif base_optimizer == 'Adam':
            self.base_optimizer = torch.optim.Adam(self.param_groups, **kwargs)
        elif base_optimizer == 'AdamW':
            self.base_optimizer = torch.optim.AdamW(self.param_groups, **kwargs)

        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step_on_data(self, data):
        adv_data = copy.deepcopy(data)
        inputs = data['inputs']

        adv_images = (inputs + self.gamma * inputs.grad.sign()).detach_()
        eta = torch.clamp(adv_images - inputs, min=-self.eps, max=self.eps)
        images = self.img_transform[0](
            torch.clamp(self.img_transform[1](self.ori_images + eta), min=0, max=255).detach_())

        adv_data = copy.deepcopy(data)
        inputs = data['inputs']

        grad_norm = inputs.grad.norm(p=2)
        scale = self.gamma / (grad_norm + 1e-12)
        e_d = inputs.grad * scale.to(inputs)

        adv_data['inputs'] = (data['inputs'] + e_d).detach_()
        return adv_data

    @torch.no_grad()
    def first_step_on_weight(self):
        # grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.dim() <= 1: continue
                grad_norm = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                scale = group["rho"] / (grad_norm + 1e-12)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                self.state[p]["old_p"] = p.data.clone()
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if "old_p" in self.state[p]:
        #             p.data = self.state[p]["old_p"]   # get back to "w" from "w + e(w)"
        #             self.state[p].pop("old_p")

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
