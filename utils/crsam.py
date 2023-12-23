import torch
import torch.nn as nn
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


class CRSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, gamma=0.1, beta=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(CRSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.gamma = gamma
        self.beta = beta
        self.o_l = 0
        self.w_l = 0
        self.b_l = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["original"] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)  
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["worst"] = p.grad.clone()
                p.sub_(self.state[p]["e_w"] * 2.0)
        if zero_grad: self.zero_grad() 

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["best"] = p.grad.clone()
                p.add_(self.state[p]["e_w"])

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = self.state[p]["worst"] + self.gamma * (self.state[p]["worst"] + self.state[p]["best"] - 2 * self.state[p]["original"]) + self.beta * ((self.state[p]["worst"] - self.state[p]["best"]))

        self.base_optimizer.step() 
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
