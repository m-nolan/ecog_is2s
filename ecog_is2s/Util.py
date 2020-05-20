import torch.nn as nn
from datetime import datetime
import numpy as np

# initialize model weights
def init_weights(m,w_range=(-0.08,0.08)):
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
