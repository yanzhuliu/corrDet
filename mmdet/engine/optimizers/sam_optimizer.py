import torch
from mmdet.registry import OPTIMIZERS

@OPTIMIZERS.register_module()
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

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
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                # grad_norm = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(p.device)
                # weight_norm = p.norm(p=2)
                # print(grad_norm)
                # print(' ')
                # print(weight_norm)
                # print('\n')
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]   # get back to "w" from "w + e(w)"
                self.state[p].pop("old_p")

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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
