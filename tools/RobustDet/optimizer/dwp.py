import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class DWP(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.001, alpha = 0.5, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.adaptive = adaptive
        self.rho = rho
        self.alpha = alpha
        defaults = dict(**kwargs)
        super(DWP, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def pre_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["grad_by_x0"] = p.grad.clone()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def grad_x0_lossl(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["grad_by_x0_lossl"] = p.grad.clone()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def grad_x0_lossc(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["grad_by_x0_lossc"] = p.grad.clone()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def grad_x1_lossl(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w_l"] = self.rho * (self.alpha * p.grad.detach() + (1 - self.alpha) * self.state[p]["grad_by_x0_lossl"])
            #    p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def grad_x1_lossc(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w_c"] = self.rho * (self.alpha * p.grad.detach() + (1 - self.alpha) * self.state[p]["grad_by_x0_lossc"])
           #     p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step_l(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w_l" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
                p.add_(self.state[p]["e_w_l"])  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step_c(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w_c" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
                p.add_(self.state[p]["e_w_c"])  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # min_x\in(x0,x1) loss(w,x) -> min_dx<||x1-x0|| loss(w,x)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                e_w = self.rho * (self.alpha * p.grad + (1 - self.alpha) * self.state[p]["grad_by_x0"])
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self, on_x = False, x=None):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if on_x:
            assert x != None
            norm = ((torch.abs(x) if self.adaptive else 1.0) * x.grad).norm(p=2).to(shared_device)
        else:
            norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
