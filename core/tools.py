
import torch
import torch.nn as nn

import numpy as np
import os
import random
import joblib

from torch.utils.data import Dataset


class F_EarlyStopping_val:

    def __init__(self, patience=10, path=None):
      
        self.patience = patience
        self.counter = 0
        self.val_min_acc = 0.
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):

 
        if  val_acc < self.val_min_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.val_min_acc = val_acc
            self.save_checkpoint(model)
            self.counter = 0
       
    def save_checkpoint(self, model):
    
        torch.save(model.state_dict(), self.path)



class dataset_torch(Dataset):

    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.target = torch.from_numpy(data_target)
        
    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len



def F_audio_datasets(data_name,data_path,T):

    test_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_x.npz'))
    test_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_y.npz'))
    testset = dataset_torch(test_set_x,test_set_y)
    del test_set_x,test_set_y

    train_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_x.npz'))
    train_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_y.npz'))
    trainset = dataset_torch(train_set_x,train_set_y)
    del train_set_x,train_set_y

    val_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'val_set_{T}_x.npz'))
    val_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'val_set_{T}_y.npz'))
    valset = dataset_torch(val_set_x,val_set_y)
    del val_set_x,val_set_y

    return trainset,valset,testset
    


def F_init_seed(seed):

    print('\nseed:',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def F_reset_all(net: nn.Module):
    
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()








# def F_audio_datasets(data_name,data_path,T,mode):

#     test_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_x.npz'))
#     test_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'test_set_{T}_y.npz'))
#     testset = dataset_torch(test_set_x,test_set_y)
#     del test_set_x,test_set_y

#     if mode == 'val':
#         train_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_x.npz'))
#         train_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_y.npz'))
#         trainset = dataset_torch(train_set_x,train_set_y)
#         del train_set_x,train_set_y

#         val_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'val_set_{T}_x.npz'))
#         val_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'val_set_{T}_y.npz'))
#         valset = dataset_torch(val_set_x,val_set_y)
#         del val_set_x,val_set_y

#         return trainset,valset,testset
    
#     elif mode == 'test':
#         if data_name == 'SHD':
#             train_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'ori_train_set_{T}_x.npz'))
#             train_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'ori_train_set_{T}_y.npz'))
#             trainset = dataset_torch(train_set_x,train_set_y)
#             del train_set_x,train_set_y

#         elif data_name == 'SSC':
#             train_set_x = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_x.npz'))
#             train_set_y = joblib.load(os.path.join(data_path+data_name+'/cache/', f'train_set_{T}_y.npz'))
#             trainset = dataset_torch(train_set_x,train_set_y)
#             del train_set_x,train_set_y
            
#         return trainset,testset
