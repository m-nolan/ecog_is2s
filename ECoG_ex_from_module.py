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
import pickle as pkl

import argparse

import matplotlib.pyplot as plt

# grab input arguments
parser = argparse.ArgumentParser('Trains a seq2seq network on a section of example NHP PMC ECoG data.',add_help=True)
parser.add_argument('--encoder-depth', metavar='el', type=int, default=10, help='Sequence depth of the encoder network')
parser.add_argument('--decoder-depth', metavar='dl', type=int, default=1, help='Sequence depth of the decoder network')
parser.add_argument('--batch-size', metavar='b', type=int, default=1, help='Data batch size')
parser.add_argument('--num-epochs', metavar='n', type=int, default=1, help='Number of optimization epochs')

args = parser.parse_args() # this bad boy has all the values packed into it. Nice!
print(args.encoder_depth,args.decoder_depth)

print(args.encoder_depth,args.decoder_depth)

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
    model_save_dir_path = '/Volumes/Samsung_T5/aoLab/Data/models/pyt/seq2seq/'
elif platform_name == 'linux2':
    # HYAK, baby!
    data_file_full_path = '/gscratch/stf/manolan/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.dat'
    mask_file_path = "/gscratch/stf/manolan/Data/WirelessData/Goose_Multiscale_M1/180325/001/rec001.LM1_ECOG_3.clfp.mask.pkl"
elif platform_name == 'linux':
    # google cloud, don't fail me now
    data_file_full_path = '/home/mickey/rec001.LM1_ECOG_3.clfp.dat'
    mask_file_path = '/home/mickey/rec001.LM1_ECOG_3.clfp.mask.pkl'
    model_save_dir_path = '/home/mickey/models/pyt/seq2seq/'

# make sure the output directory actually exists
if not os.path.exists(model_save_dir_path):
    os.makedirs(model_save_dir_path)

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
enc_len = args.encoder_depth
dec_len = args.decoder_depth
seq_len = enc_len+dec_len # use ten time points to predict the next time point

total_len_T = 60*60 # I just don't have that much time!
total_len_n = total_len_T*srate_in
data_idx = data_in.shape[1]//2 + np.arange(total_len_n)
print('Downsampling data from {0} to {1}'.format(srate_in,srate_down))
data_in = np.float32(sp.signal.decimate(data_in[:,data_idx],srate_in//srate_down,axis=-1))
print('Data Size:\t{}'.format(data_in.shape))

# filter dead channels
ch_rms = np.std(data_in,axis=-1)
ch_m = np.mean(ch_rms)
ch_low_lim = ch_m - 2*np.std(ch_rms)
ch_up_lim = ch_m + 2*np.std(ch_rms)
ch_idx = np.logical_and(ch_rms > ch_low_lim, ch_rms < ch_up_lim)
ch_list = np.arange(num_ch)[ch_idx]
num_ch_down = len(ch_list)
print('Num. ch. used:\t{}'.format(num_ch_down))
print('Ch. kept:\t{}'.format(ch_list))

#create data tensor
data_tensor = torch.from_numpy(sp.stats.zscore(data_in[ch_idx,:].view().transpose()))
if device == 'cuda:0':
    data_tensor.cuda()
print(data_tensor.size)
dataset = EcogDataloader.EcogDataset(data_tensor,device,seq_len) ## make my own Dataset class

idx_all = np.arange(dataset.data.shape[0])
idx_step = int(np.round(0.1*srate_down))
sample_idx = idx_all[:-seq_len:idx_step]
plot_seed_idx = np.array(0) # idx_all[20*60*srate_down] # this feeds the plotting dataloader, which should be producing the same plot on each run

# build the model, initialize
INPUT_SEQ_LEN = enc_len
OUTPUT_SEQ_LEN = dec_len # predict one output state from 10 inputs prior
INPUT_DIM = num_ch_down
OUTPUT_DIM = num_ch_down
HID_DIM = 4*num_ch_down
N_ENC_LAYERS = 1 
N_DEC_LAYERS = 1
ENC_DROPOUT = np.float32(0.5)
DEC_DROPOUT = np.float32(0.5)

enc = Encoder.Encoder_GRU(INPUT_DIM, HID_DIM, N_ENC_LAYERS, INPUT_SEQ_LEN, ENC_DROPOUT)
dec = Decoder.Decoder_GRU(OUTPUT_DIM, HID_DIM, N_DEC_LAYERS, OUTPUT_SEQ_LEN, DEC_DROPOUT)

model = Seq2Seq.Seq2Seq_GRU(enc, dec, device).to(device)
model.apply(Util.init_weights)

print(f'The model has {Util.count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# get to training!
train_frac = 0.8
test_frac = 0.2
valid_frac = 0.0
BATCH_SIZE = args.batch_size
N_EPOCHS = args.num_epochs
CLIP = 1

best_test_loss = float('inf')

train_loss = np.zeros(N_EPOCHS)
train_batch_loss = []
test_loss = np.zeros(N_EPOCHS)
test_batch_loss = []

# make figure (and a place to save it)
f = plt.figure()
ax = f.add_subplot(1,1,1)

# create training session directory
time_str = Util.time_str() # I may do well to pack this into util
session_save_path = os.path.join(model_save_dir_path,'enc{}_dec{}_nl{}_{}'.format(enc_len,dec_len,N_ENC_LAYERS,time_str))
sequence_plot_path = os.path.join(session_save_path,'example_sequence_figs')
os.makedirs(session_save_path) # no need to check; there's no way it exists yet.
os.makedirs(sequence_plot_path)

for e_idx, epoch in enumerate(range(N_EPOCHS)):

    start_time = time.time()

    # get new train/test splits
    train_loader, test_loader, _, plot_loader = EcogDataloader.genLoaders(dataset, sample_idx, train_frac, test_frac, valid_frac, BATCH_SIZE, plot_seed=plot_seed_idx)
    
    # get plotting data loader
    

    print('Training Network:')
    train_loss[e_idx], trbl_ = Training.train(model, train_loader, optimizer, criterion, CLIP)
    train_batch_loss.append(trbl_)
    print('Testing Network:')
    test_loss[e_idx], tebl_, _ = Training.evaluate(model, test_loader, criterion)
    test_batch_loss.append(tebl_)
    print('Running Figure Sequence:')
    plot_loss, plbl_, plot_data_tuple = Training.evaluate(model, plot_loader, criterion, plot_flag=True)
    if not (epoch % 10):
        # save the data for the plotting window in dict form
        plot_data_dict = {
            'src': plot_data_tuple[0],
            'trg': plot_data_tuple[1],
            'out': plot_data_tuple[2],
            'srate': srate_down,
        }
        torch.save(plot_data_dict,os.path.join(sequence_plot_path,'data_tuple_epoch{}.pt'.format(epoch)))
        # pass data to plotting function for this window
        f_eval,_ = Training.eval_plot(plot_data_dict)
        f_eval.savefig(os.path.join(sequence_plot_path,'eval_plot_epoch{}.png'.format(epoch)))

    end_time = time.time()

    epoch_mins, epoch_secs = Util.epoch_time(start_time, end_time)

    if test_loss[e_idx] < best_test_loss:
        best_test_loss = test_loss[e_idx]
        torch.save({ # this needs to be made into a model class method!
                'epoch': epoch,
                'num_epochs': N_EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss,
                'data_path': data_file_full_path,
                'train_frac': train_frac,
                'test_frac': test_frac,
                'batch_size': BATCH_SIZE,
                'encoder_length': enc_len,
                'decoder_length': dec_len,
                }, os.path.join(session_save_path,'model_checkpoint.pt'))

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss[e_idx]:.3g}')
    print(f'\t Test Loss: {test_loss[e_idx]:.3g}')
    if e_idx == 0:
        ax.plot(e_idx,train_loss[e_idx],'b.',label='train loss')
        ax.plot(e_idx,test_loss[e_idx],'r.',label='valid. loss')
        ax.legend(loc=0)
    else:
        ax.plot(e_idx,train_loss[e_idx],'b.')
        ax.plot(e_idx,test_loss[e_idx],'r.')

    # print the loss curve figure; continuously overwrite (like a fun stock ticker)
    f.savefig(os.path.join(session_save_path,'training_progress.png'))
