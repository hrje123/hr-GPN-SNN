import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")
from core import tools
from core import neurons
from core import losses
import model



def Parser():

    parser = argparse.ArgumentParser(description='adopted from spikingjelly')

    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=150, type=int, metavar='N',help='number of total epochs to run')
    parser.add_argument('-lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('-repeat', default=3, type=int, help='repeat nums')

    parser.add_argument('-data_path', default='/mnt/data1/hrwang/dataset/', type=str, help='root path of dataset')
    parser.add_argument('-data_name', default='SHD', type=str, help='dataset name')
    parser.add_argument('-out_dir', type=str, default='/mnt/data1/hrwang/output/', help='root dir for saving logs and checkpoint')

    parser.add_argument('-neuron_func', default=None, help='snn neuron')
    parser.add_argument('-loss_func', default=None, help='loss function') 
    parser.add_argument('-path_name', type=str, default=None, help='path name') 

    args = parser.parse_args()

    return args



def main_val(args,i):

    if i == 0:
        os.mkdir(os.path.join(args.out_dir, args.path_name))
        
    out_dir = os.path.join(args.out_dir, args.path_name, str(i+1))
    os.mkdir(out_dir)
    print(f'\nMake dir {out_dir}')

    print('\nval mode')
    print('\ndataset:%s  neuron:%s  loss:%s'%(args.data_name,args.neuron_func,args.loss_func))
    print('\nLR:%s  T:%d  batch:%d  epochs:%d\n'%(str(args.lr),args.T,args.batch_size,args.epochs))

    
    if args.neuron_func == 'GPN':
        neuron = neurons.GPN
    elif args.neuron_func == 'LIF':
        neuron = neurons.LIF
    elif args.neuron_func == 'RLIF':
        neuron = neurons.RLIF

    if args.loss_func == 'mean':
        loss_func = losses.CE_mean
    elif args.loss_func == 'last':
        loss_func = losses.CE_last

    device=torch.device('cuda')
    net=model.SNN_base_1_1024(T=args.T,dataset=args.data_name,single_step_neuron=neuron).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3, verbose=True)
    
    train_dataset,val_dataset,test_dataset = tools.F_audio_datasets(args.data_name,args.data_path,args.T)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=1,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,num_workers=1,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False,drop_last=False,num_workers=1,pin_memory=True)  


    start_epoch = 0
    writer = SummaryWriter(os.path.join(out_dir), purge_step=start_epoch)

    early_stopping = tools.F_EarlyStopping_val(patience=10, path=out_dir+'/checkpoint.pt')

    tools.F_reset_all(net)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_batch = 0

        for frame, label in train_loader:
         
            frame = frame.float().to(device)
            label = label.long().to(device)

            out_rec_TNO = net(frame)
    
            loss = loss_func(out_rec_TNO,label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            out_rec_NO = torch.mean(out_rec_TNO, dim=0)

            train_batch += 1
            train_samples += label.numel()
            _, idx = out_rec_NO.max(1)
            train_acc += np.sum((label == idx).cpu().numpy())
            train_loss += loss.cpu().item()
            
            tools.F_reset_all(net)
            
        train_loss /= train_batch
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        lr_scheduler.step(train_loss)   
        
        net.eval()
        val_acc = 0
        val_samples = 0
        val_batch = 0

        with torch.no_grad():
            for frame, label in val_loader:
          
                frame = frame.float().to(device)
                label = label.long().to(device)
        
                out_rec_TNO = net(frame)
    
                out_rec_NO = torch.mean(out_rec_TNO, dim=0)

                val_batch += 1
                val_samples += label.numel()
                _, idx = out_rec_NO.max(1)
                val_acc += np.sum((label == idx).cpu().numpy())

                tools.F_reset_all(net)
                
        val_acc /= val_samples
        writer.add_scalar('val_acc', val_acc, epoch)

        print("epoch:%d  train_loss:%.4f  train_acc:%.3f  val_acc:%.3f  time:%.1fs"%(epoch,train_loss,100*train_acc,100*val_acc,time.time()-start_time))

        if args.epochs - epoch < 50:
            early_stopping(val_acc, net)
            if early_stopping.early_stop:
                print("\n\nEarly stopping at %d epoch"%(epoch))
                break
        
        if args.loss_func == 'last':
            if epoch in [0,4,9,24,49]:
                torch.save(net.state_dict(), out_dir+'/model_'+str(epoch)+'.pt')

    net.load_state_dict(torch.load(out_dir+'/checkpoint.pt'))
    net.eval()
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for frame, label in test_loader:
     
            frame = frame.float().to(device)
            label = label.long().to(device)
    
            out_rec_TNO = net(frame)
            out_rec_NO = torch.mean(out_rec_TNO, dim=0)

            test_samples += label.numel()
            _, idx = out_rec_NO.max(1)
            test_acc += np.sum((label == idx).cpu().numpy())

            tools.F_reset_all(net)
            
    test_acc /= test_samples           
    
    print('\n-----------------')
    print('test acc%.3f'%(100*test_acc))
    print('-----------------\n')

    return test_acc*100



if __name__ == '__main__':

    args = Parser()

    acc = []
    for i in range(args.repeat):
        
        tools.F_init_seed(202302+i)
        acc.append(main_val(args,i))
    mean = np.mean(np.array(acc))
    std = np.std(np.array(acc))

    print('\n\n\n=========================')
    print('mean:%.2f  std:%.2f'%(mean,std))

