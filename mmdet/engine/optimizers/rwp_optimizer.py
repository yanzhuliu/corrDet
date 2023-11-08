import torch
from mmdet.registry import OPTIMIZERS
import numpy as np

@OPTIMIZERS.register_module()
class RWP(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, gamma=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, gamma=gamma, **kwargs)
        super(RWP, self).__init__(params, defaults)

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
            for p in group["params"]:
                if p.grad is None: continue

                dirt = torch.randn(p.size()).to(p)
                dirt.mul_(p.norm() / (dirt.norm() + 1e-10))
                len = torch.normal(0.0, group["gamma"], size=dirt.shape).to(p)
                e_w = len.mul_(dirt)

                # if len(p.shape) > 1:
                #     shape = p.shape
                #     shape_mul = np.prod(shape[1:])
                #     e_w = p.view(shape[0], -1).norm(dim=1, keepdim=True).repeat(1, shape_mul).view(p.shape)
                #     e_w = torch.normal(0, group["gamma"] * e_w).to(p.device)
                # else:
                #     e_w = torch.empty_like(p, device=p.device)
                #     e_w.normal_(0, group["gamma"] * (p.view(-1).norm().item() + 1e-16))

                self.state[p]["old_p"] = p.data.clone()
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" not in self.state[p]:
                    print("no old_p")
                    continue
                p.data = self.state[p]["old_p"]   # get back to "w" from "w + e(w)"
                self.state[p].pop("old_p")

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
