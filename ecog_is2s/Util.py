import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

# initialize model weights
def init_weights(m,w_range=(-0.08,0.08)): # what is this? Why am I doing this
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, w_range[0], w_range[1])

# return an total count of model parameters.
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# get epoch time stats, what a cutie
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def time_str():
    return datetime.strftime(datetime.now(),'%Y%m%d%H%M%S%f')

def center_diff(x):
    dx = np.zeros(x.shape)
    dx[:,1:-1] = (x[:,2:] - x[:,:-2])/2
    return np.float32(dx)

## dataset transforms
# local z-score transform
class local_zscore(object):
    # I do not know if initialization requires more detail in this case.
    def __init__(self,axis=0,scale=0.25):
        self.axis=axis
        self.scale=scale

    def __call__(self,sample_tuple):
        src, trg = sample_tuple
        sample = torch.cat([src,trg],dim=self.axis)
        mean = sample.mean(axis=self.axis)
        std = sample.std(axis=self.axis)
        src_z = (src-mean)*self.scale/std
        trg_z = (trg-mean)*self.scale/std
        return (src_z, trg_z)

# augment source signal with center-difference ds/dx estimate
class add_signal_diff(object):
    def __init__(self,srate=1):
        self.srate=srate

    def __call__(self,sample_tuple):
        src, trg = sample_tuple
        # compute center difference dsdt estimate
        dsrc = torch.zeros(src.shape,device=src.device)
        dsrc[1:-1,:] = (src[2:,:]-src[:-2,:])/(2*self.srate)
        dsrc[0,:] = dsrc[1,:]
        dsrc[-1,:] = dsrc[-2,:]
        src_aug = torch.cat((src,dsrc),axis=-1)
        return (src_aug, trg)
