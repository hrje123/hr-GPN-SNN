import numpy as np
import torch
import h5py
import argparse
import os
import random
import math
import tqdm
import joblib
from multiprocessing import Pool

import sys
sys.path.append("..")
from core import tools


"https://compneuro.net/"



def F_AUDIO_aug(data):

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j,:]=np.roll(data[i,j,:],shift=random.randint(-15, 15),axis=0)

    return data


def AUDIO_binary_image_readout(zip_audio):
    audio = []
    dt=1/args.T

    units=zip_audio[0]
    times=zip_audio[1]
    labels=zip_audio[2]
    for i in range(args.T):
        
        idxs = np.argwhere(times<=i*dt).flatten()
    
        vals = units[idxs]

        vals = vals[vals > 0]
        #vector = np.zeros(700)
        #vector[700-vals] = 1
        vector = np.bincount(700-vals)
        vector = np.pad(vector,(0,700-vector.shape[0]))
        times = np.delete(times,idxs)
        units = np.delete(units,idxs)
        audio.append(vector)
    audio.append(np.ones(700)*labels)
    return audio

def AUDIO_generate_dataset(file_name,T):
    fileh = h5py.File(file_name, mode='r')
    units = fileh["spikes"]["units"]
    times = fileh["spikes"]["times"]
    labels = fileh["labels"]
    len_times=len(times)

    del fileh

    zip_audio=[]
    for i in range(len(times)):
        zip_audio.append((units[i],times[i],labels[i]))
    
    del units,times,labels

    with Pool(20) as p:

        results = list(tqdm.tqdm(p.imap(AUDIO_binary_image_readout, zip_audio), total=len_times))


    results = np.array(results)

    return results[:,:-1], results[:,-1,0]


def AUDIO_datasets(file_name, save_path, T,transform=False):

    x,y=AUDIO_generate_dataset(file_name,T)

    if transform:
        x = transform(x)

    joblib.dump(x, save_path+'_x.npz')
    joblib.dump(y, save_path+'_y.npz')


def AUDIO_datasets_split(file_name, save_path, T, transform=False):

    x,y=AUDIO_generate_dataset(file_name,T)

    label_idx = [[] for i in range(20)]

    for i in range(len(y)):

        label_idx[int(y[i])].append(i)


    train_idx = []
    val_idx = []

    
    for i in range(20):
        np.random.shuffle(label_idx[i])
        pos = math.ceil(label_idx[i].__len__() * 0.85)
        train_idx.extend(label_idx[i][: pos])
        val_idx.extend(label_idx[i][pos: ])

    x_train = transform(x[train_idx])
    y_train = y[train_idx]

    x_val = x[val_idx]
    y_val = y[val_idx]

    joblib.dump(x_train, save_path[0]+'_x.npz')
    joblib.dump(y_train, save_path[0]+'_y.npz')

    joblib.dump(x_val, save_path[1]+'_x.npz')
    joblib.dump(y_val, save_path[1]+'_y.npz')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='generate datasets')

    parser.add_argument('-T', default=None, type=int, help='simulating time-steps')
    parser.add_argument('-data_path', default='/mnt/data1/hrwang/dataset/', type=str, help='root path of dataset')
    parser.add_argument('-data_name', default='SHD', type=str, help='dataset name')
   

    args = parser.parse_args()

    tools.F_init_seed(2023)

    args.train_set_pth = os.path.join(args.data_path+args.data_name+'/cache/', f'train_set_{args.T}')
    args.test_set_pth = os.path.join(args.data_path+args.data_name+'/cache/', f'test_set_{args.T}')
    args.val_set_pth = os.path.join(args.data_path+args.data_name+'/cache/', f'val_set_{args.T}')
    if not os.path.exists(os.path.join(args.data_path+args.data_name+'/cache/')):
        os.makedirs(os.path.join(args.data_path+args.data_name+'/cache/'))
    
    
    if args.data_name == 'SHD':
        
        AUDIO_datasets(file_name=args.data_path+args.data_name+'/origin/shd_test.h5',save_path=args.test_set_pth,T=args.T)
        AUDIO_datasets_split(file_name=args.data_path+args.data_name+'/origin/shd_train.h5',save_path=(args.train_set_pth,args.val_set_pth),T=args.T,transform=F_AUDIO_aug)

    elif args.data_name == 'SSC':
        
        AUDIO_datasets(file_name=args.data_path+args.data_name+'/origin/ssc_train.h5',save_path=args.train_set_pth,T=args.T,transform=F_AUDIO_aug)
        AUDIO_datasets(file_name=args.data_path+args.data_name+'/origin/ssc_valid.h5',save_path=args.val_set_pth,T=args.T)
        AUDIO_datasets(file_name=args.data_path+args.data_name+'/origin/ssc_test.h5',save_path=args.test_set_pth,T=args.T)

