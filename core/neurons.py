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


class LIF(BaseNode):
    def __init__(self, tau: float = 5., v_threshold: float = 0.5, v_reset: float = 0., input_channels:int = None, 
                surrogate_function: Callable = surrogate.ATan()):
        
        super().__init__(v_threshold, v_reset, surrogate_function)
        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau


class RLIF(MemoryModule):
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

