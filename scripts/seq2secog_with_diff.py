#!/usr/bin/env python
# coding: utf-8

from aopy import datareader, datafilter
from ecog_is2s import EcogDataloader, Training, Encoder, Decoder, Seq2Seq, Util

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
parser.add_argument('--num-layers', metavar='nl', type=int, default=1, help='Number of layers in each RNN block')

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

## create dataset object from file
srate = srate_in
# data_in = np.double(data_in[:,:120*srate])
enc_len = args.encoder_depth
dec_len = args.decoder_depth
seq_len = enc_len+dec_len # use ten time points to predict the next time point

total_len_T = 1*60 # I just don't have that much time!
total_len_n = total_len_T*srate_in
total_len_n_down = total_len_T*srate_down
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

#augment data with derivative estimates
data_aug = np.zeros((2*num_ch_down,total_len_n_down),dtype=np.float32)
data_aug[:num_ch_down,:] = data_in[ch_idx,:]
data_aug[num_ch_down:,:] = Util.center_diff(data_in[ch_idx,:])/srate_down

#create data tensor
data_rail = np.max(np.abs(data_in.reshape(-1)))
# normalization = 'zscore'
normalization = 'tanh'
if normalization is 'max':
    data_tensor = torch.from_numpy(data_aug[:,1:-1].view().T/data_rail)
elif normalization is 'zscore':
    # for nominally gaussian data distributions, this will get ~99% of data points in (-1, 1)
    data_tensor = torch.from_numpy(sp.stats.zscore(data_aug[:,1:-1].view().T)/5)
elif normalization is 'tanh':
    data_tensor = torch.from_numpy(np.tanh(sp.stats.zscore(data_aug[:,1:-1].view().T)/3))

if device == 'cuda:0':
    data_tensor.cuda()
print(data_tensor.size)
dataset = EcogDataloader.EcogDataset(data_tensor,device,seq_len) ## make my own Dataset class

idx_all = np.arange(dataset.data.shape[0])
idx_step = int(np.round(0.1*srate_down))
sample_idx = idx_all[:-seq_len:idx_step]
n_plot_seed = 3
n_plot_step = seq_len // idx_step
plot_seed_idx = np.arange(0,n_plot_seed*n_plot_step,n_plot_step)

# build the model, initialize
INPUT_SEQ_LEN = enc_len # number of samples to feed to encoder
OUTPUT_SEQ_LEN = dec_len # number of samples to predict with decoder
INPUT_DIM = 2*num_ch_down
OUTPUT_DIM = num_ch_down
HID_DIM = 4*num_ch_down
N_LAYER = args.num_layers
N_ENC_LAYERS = N_LAYER
N_DEC_LAYERS = N_LAYER
ENC_DROPOUT = np.float32(0.5)
DEC_DROPOUT = np.float32(0.5)
LEARN_RATE = 0.01 # default ADAM: 0.001
WEIGHT_RANGE = (-0.2,0.2) # ignore for now; not sure how to worm this through


enc = Encoder.Encoder_GRU(INPUT_DIM, HID_DIM, N_ENC_LAYERS, INPUT_SEQ_LEN, ENC_DROPOUT)
dec = Decoder.Decoder_GRU(OUTPUT_DIM, HID_DIM, N_DEC_LAYERS, OUTPUT_SEQ_LEN, DEC_DROPOUT)

model = Seq2Seq.Seq2Seq_GRU(enc, dec, device).to(device)
model.apply(Util.init_weights)

print(f'The model has {Util.count_parameters(model):,} trainable parameters')

weight_reg = 20.
optimizer = optim.Adam(model.parameters(),lr=LEARN_RATE,weight_decay=weight_reg)
# criterion = nn.L1Loss(reduction='mean')
loss_obj = 'MSE' #L1, L2, see training.py:ECOGLoss()
criterion = Training.ECOGLoss(sum_axis=1,objective=loss_obj)

# get to training!
train_frac = 0.8
test_frac = 0.2
valid_frac = 0.0
BATCH_SIZE = args.batch_size
N_EPOCHS = args.num_epochs
CLIP = 1. # this the maximum norm of the whole parameter gradient.
TFR = 0.0 # no teacher forcing! Anything it's learning is all on its own
RAND_SAMP = False

best_test_loss = float('inf')

train_loss = np.zeros(N_EPOCHS)
train_batch_loss = []
test_loss = np.zeros(N_EPOCHS)
test_batch_loss = []

# create training session directory
time_str = Util.time_str() # I may do well to pack this into util
session_save_path = os.path.join(model_save_dir_path,'enc{}_dec{}_nl{}_nep{}_{}'.format(enc_len,dec_len,N_ENC_LAYERS,N_EPOCHS,time_str))
sequence_plot_path = os.path.join(session_save_path,'example_sequence_figs')
os.makedirs(session_save_path) # no need to check; there's no way it exists yet.
os.makedirs(sequence_plot_path)
print('saving session data to:\t{}'.format(session_save_path))
# save a histogram of the data distribution; allowing you to check
f,ax = plt.subplots(2,1,figsize=(8,4))
ax[0].hist(dataset.data[:,:num_ch_down].reshape(-1),100,density=True,label='ECoG')
ax[1].hist(dataset.data[:,num_ch_down:].reshape(-1),100,density=True,label='dECoG')
f.savefig(os.path.join(session_save_path,'norm_data_hist.png'))

# make figure (and a place to save it)
f_loss = plt.figure()
ax_loss = f_loss.add_subplot(1,1,1)

for e_idx, epoch in enumerate(range(N_EPOCHS)):

    start_time = time.time()

    # get new train/test splits
    # note: add switch to genLoaders to allow for fixed/random sampling
    train_loader, test_loader, _, plot_loader = EcogDataloader.genLoaders(dataset, sample_idx, train_frac, test_frac, valid_frac, BATCH_SIZE, rand_samp=RAND_SAMP, plot_seed=plot_seed_idx)

    print('Training Network:')
    _, trbl_ = Training.train(model, train_loader, optimizer, criterion, CLIP, TFR)
    train_batch_loss.append(trbl_)
    train_loss[e_idx] = np.mean(trbl_) # this is the plotted training loss
    print('Testing Network:')
    _, tebl_, _ = Training.evaluate(model, test_loader, criterion)
    # test_batch_loss.append(tebl_)
    test_loss[e_idx] = np.mean(tebl_) # this is the plotted test loss
    print('Running Figure Sequence:')
    plot_loss, plbl_, plot_data_list = Training.evaluate(model, plot_loader, criterion, plot_flag=True)
    if not (epoch % 10):
        print('Saving estimate plots:')
        # save the data for the plotting window in dict form
        epoch_plot_path = os.path.join(sequence_plot_path,'epoch{}'.format(epoch))
        os.makedirs(epoch_plot_path)
        torch.save(model.state_dict(),os.path.join(epoch_plot_path,'model_epoch{}.pt'.format(epoch)))
        c_list = ['b','r']
        for k in range(len(plot_data_list)):
            c_output = c_list[k//n_plot_seed] # blue for training windows, red for testing windows
            plot_data_dict = {
                'src': plot_data_list[k][0],
                'src_dx': plot_data_list[k][1],
                'trg': plot_data_list[k][2],
                'out': plot_data_list[k][3],
                'enc': plot_data_list[k][4],
                'dec': plot_data_list[k][5],
                'srate': srate_down,
                # 'state_dict': model.state_dict(), # putting this in every file is redundant
            }
            torch.save(plot_data_dict,os.path.join(epoch_plot_path,'data_tuple_epoch{}_window{}.pt'.format(epoch,k)))
            # pass data to plotting function for this window
            # for plots:
            # blue: train
            # red: test
            # black: real
            # green: encoder
            # magenta: decoder
            f_out, f_enc, f_dec, f_src = Training.eval_plot(plot_data_dict,c_output=c_output)
            # save plots in current epoch subdir
            f_out.savefig(os.path.join(epoch_plot_path,'output_plot_epoch{}_window{}.png'.format(epoch,k)))
            f_enc.savefig(os.path.join(epoch_plot_path,'encoder_plot_epoch{}_window{}.png'.format(epoch,k)))
            f_dec.savefig(os.path.join(epoch_plot_path,'decoder_plot_epoch{}_window{}.png'.format(epoch,k)))
            f_src.savefig(os.path.join(epoch_plot_path,'source_plot_epoch{}_window{}.png'.format(epoch,k)))
            [plt.close(f) for f in [f_out,f_enc,f_dec]]

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
        ax_loss.plot(e_idx,train_loss[e_idx],'b.',label='train loss')
        ax_loss.plot(e_idx,test_loss[e_idx],'r.',label='valid. loss')
        ax_loss.legend(loc=0)
    else:
        ax_loss.plot(e_idx,train_loss[e_idx],'b.')
        ax_loss.plot(e_idx,test_loss[e_idx],'r.')
    ax_loss.set_ylim(bottom=0,top=1.05*np.concatenate((train_loss,test_loss)).max())
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss: {}'.format(loss_obj))
    # print the loss curve figure; continuously overwrite (like a fun stock ticker)
    f_loss.savefig(os.path.join(session_save_path,'training_progress.png'))
    torch.save({'train_loss':train_loss,'test_loss':test_loss,},os.path.join(session_save_path,'training_progress.pt'))
