import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math
import time

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22}) # this may be a problem w/in a module?

# not sure if this belongs here
def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0
    batch_loss = []

    enc_len = model.encoder.seq_len
    dec_len = model.decoder.seq_len

    for i, batch in enumerate(iterator):
        if np.mod(i+1,1000) == 0:
            print(i,len(iterator))
        src = batch[:,:enc_len,:]
        trg = batch[:,enc_len:enc_len+dec_len,:]
        if dec_len == 1:
            trg = trg.unsqueeze(1)

        optimizer.zero_grad()
#         breakpoint()
        output = model(src, trg)

        #trg = [batch size, trg len, output dim]
        #output = [batch size, trg len, output dim]

        output_dim = output.shape[-1]

#         output = output[1:].view(-1, output_dim)
#         trg = trg[1:].view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        batch_loss.append(loss.item())

#         bar.update(10*i/10000)

#         if i > 10000:
#             break

    return epoch_loss / len(iterator), np.array(batch_loss)

def evaluate(model, iterator, criterion, plot_flag=False):

    model.eval()

    epoch_loss = 0
    batch_loss = []

    enc_len = model.encoder.seq_len
    dec_len = model.decoder.seq_len

    with torch.no_grad():
#         widgets = [pb.Percentage(), progressbar.Bar()]
#         bar = pb.ProgressBar(widgets=widgets).start()
#         i = 0
        for i, batch in enumerate(iterator):

            if np.mod(i+1,1000)==0:
                print(i,len(iterator))
            src = batch[:,:enc_len,:]
            trg = batch[:,enc_len:enc_len+dec_len,:]
            if dec_len == 1:
                trg = trg.unsqueeze(1)

            output = model(src, trg, 0.) #turn off teacher forcing

            #trg = [batch size, trg len]
            #output = [batch size, trg len, output dim]

            output_dim = output.shape[-1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            batch_loss.append(loss.item())

        num_batch = i+1
        if plot_flag:
            plot_output = []
            for k in range(num_batch):
                plot_output.append((src[k,],trg[k,],output[k,]))
        else:
            plot_output = []

    # there may be a bug in the loss normalization here
    return epoch_loss / len(iterator), np.array(batch_loss), plot_output

def eval_plot(plot_dict,figsize=(10.5,8),n_pca=0):
    # compute PCA dims from catted src/trg data
    # make sure to push this frame back to the cpu!
    src = plot_dict['src'].cpu()#.squeeze(dim=0)
    trg = plot_dict['trg'].cpu()#.squeeze(dim=0)
    out = plot_dict['out'].cpu()#.squeeze(dim=0)
    print(src.shape,trg.shape,out.shape)
    # n_win = src.shape(0) # each plotted window will be a separate batch here
    n_t = trg.shape(0)
    n_ch = trg.shape(-1)
    n_r = 8
    n_c = 8
    # if n_pca > 0:
    #     # change this to a 'do nothing' switch; PCA isn't proven yet
    #     print(target_red.shape,output_red.shape)
    #     target_n = target_red.shape[0]
    #     plot_t = np.arange(target_n)/plot_dict['srate']
    # #     print(plot_t)
    #     f,ax = plt.subplots(n_pca,1,figsize=(figsize[0],n_pca*figsize[1]))
    #     if n_pca == 1:
    #         ax = [ax]
    #     for n in range(n_pca):
    #         ax[n].plot(plot_t,target_red[:,n],label='trg_{}'.format(n))
    # #         print(target_red[:,n])
    #         ax[n].plot(plot_t,output_red[:,n],label='out_{}'.format(n))
    # #         print(output_red[:,n])
    #         ax[n].legend(loc=0)
    #         ax[n].set_xlabel('time (s)')
    #         ax[n].set_ylabel('PC{}'.format(n))
    # elif:
    target_n = target_red.shape[0]
    plot_t = np.arange(target_n)/plot_dict['srate']
    f,ax = plt.subplots(n_r,n_c,figsize=figsize)
    for n in range(n_ch):
        r_idx = n // n_c
        c_idx = c % n_c
        ax[r_idx,c_idx].plot(plot_t,trg[:,n])
        ax[r_idx,c_idx].plot(plot_t,out[:,n])
        ax[r_idx,c_idx].legend('{}'.format(n))
    ax[r_idx,0].set_xlabel('time (s)')

    return f, ax

# silly tool to format epoch computation times
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# def plot_example_sequence(data_tuple,figsize=())
