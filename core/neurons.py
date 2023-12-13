import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from abc import abstractmethod
from typing import Callable
import math

import sys
sys.path.append("..")
from core import surrogate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MemoryModule(nn.Module):
    def __init__(self):
      
        super().__init__()

        self.train_param = True

        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        
        assert not hasattr(self, name), f'{name} has been set as a member variable'

        self._memories[name] = value

        self._memories_rv[name] = copy.deepcopy(value)

    def reset_one(self,key):
        
        self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def reset(self):
        
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories


        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
    
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)

        for key, value in self._memories_rv.items():
            if isinstance(value, torch.Tensor):
                self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class BaseNode(MemoryModule):
    def __init__(self, v_threshold: float = 0.5, v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__()

        self.register_memory('v', v_reset)
        self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        
        raise NotImplementedError

    def neuronal_fire(self):
        
        self.spike = self.surrogate_function(self.v - self.v_threshold)
    
    def neuronal_reset(self):

        spike = self.spike.detach()
        
        self.v = (1. - spike) * self.v + spike * self.v_reset

    def forward(self, x: torch.Tensor):
        
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

class IF(BaseNode):
    def __init__(self, v_threshold: float = 0.5, v_reset: float = 0., input_channels:int = None, 
                surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__(v_threshold, v_reset, surrogate_function)

    def neuronal_charge(self, x: torch.Tensor):
        
        self.v = self.v + x


class LIF(BaseNode):
    def __init__(self, tau: float = 5., v_threshold: float = 0.5, v_reset: float = 0., input_channels:int = None, 
                surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__(v_threshold, v_reset, surrogate_function)
        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau


class RLIF(MemoryModule):
    """
    Cuba-LIF
    """
    def __init__(self, alpha: float = 0.2, beta: float = 0.2, v_threshold: float = 0.5, v_reset: float = 0., input_channels:int = 1024, 
                 surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__()

        self.register_memory('v', v_reset)
        self.register_memory('i', 0.)
        self.register_memory('spike', 0.)
        
        self.alpha = alpha
        self.beta = beta
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.surrogate_function = surrogate_function

        self.fc_U = nn.Linear(input_channels, input_channels)

    def neuronal_charge(self, x: torch.Tensor):

        if type(self.v) == float:
            self.v = torch.zeros(x.shape).to(device)
        if type(self.i) == float:
            self.i = torch.zeros(x.shape).to(device)
        if type(self.spike) == float:    
            self.spike = torch.zeros(x.shape).to(device)

        self.i = self.alpha * self.i + x + self.fc_U(self.spike)
        
        self.v = self.beta * self.v + (1-self.beta) * self.i

    def neuronal_fire(self):

        self.spike = self.surrogate_function(self.v - self.v_threshold)
    
    def neuronal_reset(self):

        spike = self.spike.detach()
        self.v = (1. - spike) * self.v + spike * self.v_reset

    def forward(self, x: torch.Tensor):
        
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()

        return self.spike


class ArchAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0.5)
        return output.float()
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class GLIF(nn.Module):
    '''
    GLIF 
    https://github.com/Ikarosy/Gated-LIF/tree/master
    '''
    def __init__(self, T, **kwargs):
        super(GLIF, self).__init__()
        self.T = T
        self.soft_mode = False
        self.static_gate = True
        self.static_param = False
        self.time_wise = True
        self.param = [0.2, 0.5, 0.5/(self.T*2), 0.5]
        #c
        self.alpha, self.beta, self.gamma = [nn.Parameter(- math.log(1 / ((i - 0.5)*0.5+0.5) - 1) * torch.ones(1024, dtype=torch.float))
                                                 for i in [0.6,0.8,0.6]]

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(1024, dtype=torch.float))
                              for i in self.param[:-1]]
        self.reVth = nn.Parameter(- math.log(1 / self.param[1] - 1) * torch.ones(1024, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, 1024), dtype=torch.float))
                                   for i in self.param[3:]][0]

    def forward(self, x): #t, b, c,
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            

            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1).clone().detach().sigmoid(), self.beta.view(1, -1).clone().detach().sigmoid(), self.gamma.view(1, -1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1).clone().detach().gt(0.).float(), self.beta.view(1, -1).clone().detach().gt(0.).float(), self.gamma.view(1, -1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1).sigmoid(), self.beta.view(1, -1).sigmoid(), self.gamma.view(1, -1).sigmoid()
            else:
                al, be, ga = ArchAct.apply(self.alpha.view(1, -1).sigmoid()), ArchAct.apply(self.beta.view(1, -1).sigmoid()), ArchAct.apply(self.gamma.view(1, -1).sigmoid())

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :]) + \
                  I_t1 - (1 - ga) * reVth[None, :] * o_t_n1.clone()
        o_t1_n1 = surrogate.ATan()(u_t1_n1 - Vth[None, :])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))


class PLIF(BaseNode):
    """
    https://arxiv.org/abs/2007.05785
    """
    def __init__(self, init_tau: float = 5., v_threshold: float = 0.5, v_reset: float = 0., 
                surrogate_function: Callable = surrogate.ATan()):
       

        super().__init__(v_threshold, v_reset, surrogate_function)
        
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def neuronal_charge(self, x: torch.Tensor):
       
        self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()


class GPN(MemoryModule):
    def __init__(self, input_channels: int = 1024, surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__()

        self.register_memory('v', 0.)
        self.register_memory('spike', 0.)
        self.register_memory('v_reset',0.)

        self.surrogate_function = surrogate_function
        self.input_channels = input_channels

        self.ln = nn.Linear(in_features=2*self.input_channels,
                              out_features=4*self.input_channels
                             )

    def forward(self, x: torch.Tensor):
        
        if type(self.spike) == float:
      
            if self.training:
                self.v_reset = torch.normal(mean=0.,std=0.05,size=x.shape).to(device)
            else:
                self.v_reset = torch.zeros(x.shape).to(device)
                
            self.v = self.v_reset
            self.spike = torch.zeros(x.shape).to(device)

        combined = torch.cat([self.v, x], dim=1)

        combined_ln = self.ln(combined)

        cc_f, cc_i, cc_b, cc_t = torch.split(combined_ln, self.input_channels, dim=1)
        
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        b = torch.tanh(cc_b)
        t = torch.sigmoid(cc_t)
   
        v_hidden = f * self.v + i * x

        self.spike = self.surrogate_function(v_hidden - t)
        
        spike_reset = self.spike.detach()
        self.v = (1. - spike_reset) * v_hidden + spike_reset * self.v_reset + b

        return self.spike

