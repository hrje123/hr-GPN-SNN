import torch
import torch.nn as nn

import math


def heaviside(x: torch.Tensor):
 
    return (x >= 0).to(x)



class ATan_base(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None

class ATan(nn.Module):
    def __init__(self, alpha=2.0):
       
        super().__init__()

        self.alpha=alpha

    def forward(self,x):
        
        return ATan_base.apply(x, self.alpha)
   
