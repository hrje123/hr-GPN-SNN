import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

from core import methods

#snn
class SNN_base_1_1024(nn.Module):
    def __init__(self, T, dataset, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        if dataset == 'SHD':
            self.num_labels = 20
        elif dataset == 'SSC':
            self.num_labels = 35

        self.T = T

        self.fc1 = nn.Linear(700, 1024)
        self.sn1 = single_step_neuron(**kwargs)
        self.dp1 = methods.Dropout(0.25)
        self.fc2 = nn.Linear(1024, self.num_labels)
    
    def forward(self, x_input: torch.Tensor):
        
        out_rec=[]

        x_input = x_input.permute(1, 0, 2)

        for t in range(self.T):

            x = self.fc1(x_input[t])
            x = self.sn1(x)
            x = self.dp1(x)
            x = self.fc2(x)

            out_rec.append(x)

        out_rec_TBdims = torch.stack(out_rec, dim=0)
     
        return out_rec_TBdims

#snn
class SNN_base_3_1024(nn.Module):
    def __init__(self, T, dataset, single_step_neuron: callable = None, **kwargs):
        super().__init__()

        if dataset == 'SHD':
            self.num_labels = 20
        elif dataset == 'SSC':
            self.num_labels = 35

        self.T = T

        self.fc1 = nn.Linear(700, 1024)
        self.sn1 = single_step_neuron(**kwargs)
        self.dp1 = methods.Dropout(0.25)

        self.fc2 = nn.Linear(1024, 1024)
        self.sn2 = single_step_neuron(**kwargs)
        self.dp2 = methods.Dropout(0.25)

        self.fc3 = nn.Linear(1024, 1024)
        self.sn3 = single_step_neuron(**kwargs)
        self.dp3 = methods.Dropout(0.25)

        self.fc4 = nn.Linear(1024, self.num_labels)
   

    def forward(self, x_input: torch.Tensor):
        
        out_rec=[]

        x_input = x_input.permute(1, 0, 2)


        for t in range(self.T):

            x = self.fc1(x_input[t])
            x = self.sn1(x)
            x = self.dp1(x)

            x = self.fc2(x)
            x = self.sn2(x)
            x = self.dp2(x)

            x = self.fc3(x)
            x = self.sn3(x)
            x = self.dp3(x)

            x = self.fc4(x)

            out_rec.append(x)

        out_rec_TBdims = torch.stack(out_rec, dim=0)
     
        return out_rec_TBdims



