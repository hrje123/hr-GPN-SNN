import torch
import torch.nn as nn
import torch.nn.functional as F



def CE_mean(out_rec_TNO,label):

    out_rec_NO = torch.mean(out_rec_TNO, dim=0)

    loss = nn.CrossEntropyLoss()(out_rec_NO,label)

    return loss


def CE_last(out_rec_TNO,label):

    out_rec_NO = out_rec_TNO[-1]

    loss = nn.CrossEntropyLoss()(out_rec_NO,label)

    return loss


