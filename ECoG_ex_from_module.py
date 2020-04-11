#!/usr/bin/env python
# coding: utf-8

from aopy import datareader, datafilter
from ecog_is2s import EcogDataloader, Training
from ecog_is2s.model import Encoder, Decoder, Seq2Seq
from ecog_is2s.model import Util

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler, BatchSampler, SubsetRandomSampler
from torch.utils.data import TensorDataset, random_split

import spacy
import numpy as np
import pandas as pd
# import sklearn
import scipy as sp

import random
import math
import time

# import progressbar as pb
import datetime
import os
import sys

import matplotlib.pyplot as plt

import pickle as pkl


# seed RNG for pytorch/np
SEED = 5050
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set device - CUDA if you've got it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('mounting to device: {}'.format(device))

# load data
platform_name = sys.platform
if platform_name == 'darwin':
    # local machine
    data_file_full_path = '/Volumes/Samsung_T5/aoLab/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.dat'
    mask_file_path = "/Volumes/Samsung_T5/aoLab/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.mask.pkl"
elif platform_name == 'linux2':
    # HYAK, baby!
    data_file_full_path = '/gscratch/stf/manolan/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.dat'
    mask_file_path = "/gscratch/stf/manolan/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.mask.pkl"
elif platform_name == 'linux':
    # google cloud, don't fail me now
    data_file_full_path = '/home/mickey/rec001.LM1_ECOG_3.clfp.dat'
    mask_file_path = '/home/mickey/rec001.LM1_ECOG_3.clfp.mask.pkl'

data_in, data_param, data_mask = datareader.load_ecog_clfp_data(data_file_name=data_file_full_path)
srate_in= data_param['srate']
num_ch = data_param['num_ch']
# we already found the appropriate data masks, so just load them in
with open(mask_file_path, 'rb') as f:
    mask_data = pkl.load(f)
hf_mask = mask_data["hf"]
sat_mask = mask_data["sat"]

# mask data array, remove obvious outliers
data_in[:,np.logical_or(hf_mask,sat_mask)] = 0.

# downsample data
srate_down = 250

# create dataset object from file
srate = srate_in
# data_in = np.double(data_in[:,:120*srate])
enc_len = 10
dec_len = 1
seq_len = enc_len+dec_len # use ten time points to predict the next time point

total_len_T = 1*60 # I just don't have that much time!
total_len_n = total_len_T*srate_in
data_idx = data_in.shape[1]//2 + np.arange(total_len_n)
print('Downsampling data from {0} to {1}'.format(srate_in,srate_down))
data_in = np.float32(sp.signal.decimate(data_in[:,data_idx],srate_in//srate_down,axis=-1))

data_tensor = torch.from_numpy(sp.stats.zscore(data_in.view().transpose()))
if device == 'cuda:0':
    data_tensor.cuda()
print(data_tensor.size)
dataset = EcogDataloader.EcogDataset(data_tensor,device,seq_len) ## make my own Dataset class

idx_all = np.arange(dataset.data.shape[0])
sample_idx = idx_all[:-seq_len]

# build the model, initialize
INPUT_SEQ_LEN = enc_len
OUTPUT_SEQ_LEN = dec_len # predict one output state from 10 inputs prior
INPUT_DIM = num_ch
OUTPUT_DIM = num_ch
HID_DIM = num_ch
N_ENC_LAYERS = 1 
N_DEC_LAYERS = 1
ENC_DROPOUT = np.float32(0.5)
DEC_DROPOUT = np.float32(0.5)

enc = Encoder.Encoder_GRU(INPUT_DIM, HID_DIM, N_ENC_LAYERS, ENC_DROPOUT)
dec = Decoder.Decoder_GRU(OUTPUT_DIM, HID_DIM, N_DEC_LAYERS, DEC_DROPOUT)

model = Seq2Seq.Seq2Seq_GRU(enc, dec, device).to(device)
model.apply(Util.init_weights)

print(f'The model has {Util.count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# get to training!
train_frac = 0.8
test_frac = 0.2
valid_frac = 0.0
BATCH_SIZE = 10
N_EPOCHS = 50
CLIP = 1

best_test_loss = float('inf')

train_loss = np.zeros(N_EPOCHS)
train_batch_loss = []
test_loss = np.zeros(N_EPOCHS)
test_batch_loss = []

# make figure (and a place to save it)
f = plt.figure()
ax = f.add_subplot(1,1,1)


for e_idx, epoch in enumerate(range(N_EPOCHS)):
    
    start_time = time.time()
    
    # get new train/test splits
    train_loader, test_loader, _ = EcogDataloader.genLoaders(dataset, sample_idx, train_frac, test_frac, valid_frac, BATCH_SIZE)
    
    print('Training Network:')
    train_loss[e_idx], trbl_ = Training.train(model, train_loader, optimizer, criterion, CLIP)
    train_batch_loss.append(trbl_)
    print('Testing Network:')
    test_loss[e_idx], tebl_ = Training.evaluate(model, test_loader, criterion)
    test_batch_loss.append(tebl_)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = Util.epoch_time(start_time, end_time)
    
    if test_loss[e_idx] < best_test_loss:
        best_test_loss = test_loss[e_idx]
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss[e_idx]:.3g}')
    print(f'\t Test Loss: {test_loss[e_idx]:.3g}')
    
    ax.plot(e_idx,train_loss[e_idx],'b.')
    ax.plot(e_idx,test_loss[e_idx],'r.')
    
    # print the figure; continuously overwrite (like a fun stock ticker)
    f.savefig('training_progress.png')

# save model

