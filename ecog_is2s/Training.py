import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss, l1_loss
from torch.nn.modules.loss import _Loss, _WeightedLoss

import numpy as np

import math
import time

import matplotlib.pyplot as plt

# not sure if this belongs here
def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):

    model.train()

    epoch_loss = 0
    batch_loss = []

    enc_len = model.encoder.seq_len
    dec_len = model.decoder.seq_len
    n_ch = model.decoder.output_dim

    for i, batch in enumerate(iterator):
        if np.mod(i+1,1000) == 0:
            print(i,len(iterator))
        src = batch[:,:enc_len,:]
        trg = batch[:,enc_len:enc_len+dec_len,:n_ch] # only train to
        if dec_len == 1:
            trg = trg.unsqueeze(1)

        optimizer.zero_grad()
        output, _, _ = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        #trg = [batch size, trg len, output dim]
        #output = [batch size, trg len, output dim]

        loss = criterion(output, trg)
        loss.backward()

        # clipping the gradient norm may not be very helpful...
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        batch_loss.append(loss.item())

#         bar.update(10*i/10000)

#         if i > 10000:
#             break

    return epoch_loss, np.array(batch_loss)

def evaluate(model, iterator, criterion, plot_flag=False):

    model.eval()

    epoch_loss = 0
    batch_loss = []

    enc_len = model.encoder.seq_len
    dec_len = model.decoder.seq_len

    n_ch = model.decoder.output_dim # rename to *_dim?
    n_h_ch = model.encoder.hid_dim

    batch_size = iterator.batch_size

    with torch.no_grad():
#         widgets = [pb.Percentage(), progressbar.Bar()]
#         bar = pb.ProgressBar(widgets=widgets).start()
#         i = 0
        if plot_flag:
            src_ = torch.zeros(len(iterator),enc_len,n_ch)
            src_dx_ = torch.zeros(len(iterator),enc_len,n_ch)
            trg_ = torch.zeros(len(iterator),dec_len,n_ch)
            out_ = torch.zeros(len(iterator),dec_len,n_ch)
            enc_ = torch.zeros(len(iterator),enc_len,n_h_ch)
            dec_ = torch.zeros(len(iterator),dec_len,n_h_ch)

        for i, batch in enumerate(iterator):

            if np.mod(i+1,1000)==0:
                print(i,len(iterator))
            src = batch[:,:enc_len,:]
            trg = batch[:,enc_len:enc_len+dec_len,:n_ch]
            if dec_len == 1:
                trg = trg.unsqueeze(1)

            output, enc_state, dec_state = model(src, trg, teacher_forcing_ratio=0.) #turn off teacher forcing
            if plot_flag:
                src_[i,] = src[:,:,:n_ch]
                if src.shape[-1] > n_ch:
                    src_dx_[i,] = src[:,:,n_ch:]
                else:
                    src_dx_[i,] = 0. # works?
                trg_[i,] = trg
                out_[i,] = output
                enc_[i,] = enc_state
                dec_[i,] = dec_state

            #trg = [batch size, trg len]
            #output = [batch size, trg len, output dim]

            output_dim = output.shape[-1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            batch_loss.append(loss.item())

        plot_output = []
        if plot_flag:
            # print('num batch:\t{}'.format(num_batch))
            # print(src.size(),trg.size(),output.size())
            for k in range(len(src_)):
                plot_output.append((src_[k,],src_dx_[k,],trg_[k,],out_[k,],enc_[k,],dec_[k,]))

    # there may be a bug in the loss normalization here
    return epoch_loss, np.array(batch_loss), plot_output

def eval_plot(plot_dict,figsize=(10.5,8),n_pca=0,c_output='b',c_target='k',c_encoder='g',c_decoder='m'):
    # compute PCA dims from catted src/trg data
    # make sure to push this frame back to the cpu!
    src = plot_dict['src'].cpu().numpy()#.squeeze(dim=0)
    src_dx = plot_dict['src_dx'].cpu().numpy()
    trg = plot_dict['trg'].cpu().numpy()#.squeeze(dim=0)
    out = plot_dict['out'].cpu().numpy()#.squeeze(dim=0)
    enc = plot_dict['enc'].cpu().numpy()#.squeeze(dim=0)
    dec = plot_dict['dec'].cpu().numpy()#.squeeze(dim=0)
    srate = plot_dict['srate']
    # print(src.shape,trg.shape,out.shape)
    # n_win = src.shape(0) # each plotted window will be a separate batch here
    n_t_in = src.shape[0]
    n_t_out = trg.shape[0]
    n_ch = trg.shape[-1]
    n_c_max = 8
    n_c = np.min((8,n_ch))
    n_r = n_ch//n_c + 2 # give an extra row for the scale bar

    # get scale bar references
    lp10 = lambda x: 10**np.floor(np.log10(x))
    dom = lambda a: np.max(a.reshape(-1))-np.min(a.reshape(-1))
    t_in = n_t_in/srate
    t_out = n_t_out/srate
    bar_t_in = lp10(t_in)
    bar_t_out = lp10(t_out)
    a_src = dom(src)
    a_trg = dom(trg)
    a_out = dom(out)
    a_enc = dom(enc)
    a_dec = dom(dec)
    bar_a_src = lp10(a_src)
    bar_a_trg = lp10(a_trg)
    bar_a_out = lp10(a_out)
    bar_a_enc = lp10(a_enc)
    bar_a_dec = lp10(a_dec)

    plot_t_in = np.arange(n_t_in)/plot_dict['srate']
    plot_t_out = np.arange(n_t_out)/plot_dict['srate']
    plt.rcParams.update({'font.size': 8}) # this may be a problem w/in a module?

    # plot source, dsdt
    f_source,ax = plt.subplots(n_r,n_c,figsize=figsize,sharex=True,sharey=True)
    if len(ax.shape) < 2:
        ax = ax[:,None]
    for n in range(n_r*n_c):
        r_idx = n // n_c
        c_idx = n % n_c
        if n < n_ch:
            ax[r_idx,c_idx].plot(plot_t_in,src[:,n],color=c_target)
            if src_dx.shape[-1]:
                ax[r_idx,c_idx].plot(plot_t_in,src_dx[:,n],color=c_target,linestyle='--')
            ax[r_idx,c_idx].get_xaxis().set_ticks([])
            ax[r_idx,c_idx].get_yaxis().set_ticks([])
            # ax[r_idx,c_idx].legend('{}'.format(n))
        plt.sca(ax[r_idx,c_idx])
        plt.box(on=False)
        plt.xticks([])
        plt.yticks([])
    # add time, amplitude references
    x_min, x_max, y_min, y_max = ax[r_idx,0].axis()
    x_w_in = bar_t_out/(x_max-x_min)
    y_w = bar_a_src/(y_max-y_min)
    # amplitude scale bar
    ax[r_idx,0].axvline(x_min,ymin=0,ymax=y_w,linewidth=3,color='k')
    ax[r_idx,0].text(x_min,y_min+bar_a_src/2,"{} ".format(bar_a_src),
                     horizontalalignment='right',verticalalignment='center')
    # time scale bar
    ax[r_idx,0].axhline(y_min,xmin=0,xmax=x_w_in,linewidth=3,color='k')
    ax[r_idx,0].text(x_min+bar_t_in/2,y_min,"{} s".format(bar_t_in),
                     horizontalalignment='center',verticalalignment='top')

    # plot target v. output
    f_target_v_out,ax = plt.subplots(n_r,n_c,figsize=figsize,sharex=True,sharey=True)
    if len(ax.shape) < 2:
        ax = ax[:,None]
    for n in range(n_r*n_c):
        r_idx = n // n_c
        c_idx = n % n_c
        if n < n_ch:
            ax[r_idx,c_idx].plot(plot_t_out,trg[:,n],color=c_target)
            ax[r_idx,c_idx].plot(plot_t_out,out[:,n],color=c_output)
            ax[r_idx,c_idx].get_xaxis().set_ticks([])
            ax[r_idx,c_idx].get_yaxis().set_ticks([])
            # ax[r_idx,c_idx].legend('{}'.format(n))
        plt.sca(ax[r_idx,c_idx])
        plt.box(on=False)
        plt.xticks([])
        plt.yticks([])
    # add time, amplitude references
    x_min, x_max, y_min, y_max = ax[r_idx,0].axis()
    x_w_in = bar_t_out/(x_max-x_min)
    bar_a_trg_out_max = np.max((bar_a_trg,bar_a_out))
    y_w = bar_a_trg_out_max/(y_max-y_min)
    # amplitude scale bar
    ax[r_idx,0].axvline(x_min,ymin=0,ymax=y_w,linewidth=3,color='k')
    ax[r_idx,0].text(x_min,y_min+bar_a_trg_out_max/2,"{} ".format(bar_a_trg_out_max),
                     horizontalalignment='right',verticalalignment='center')
    # time scale bar
    ax[r_idx,0].axhline(y_min,xmin=0,xmax=x_w_in,linewidth=3,color='k')
    ax[r_idx,0].text(x_min+bar_t_out/2,y_min,"{} s".format(bar_t_out),
                     horizontalalignment='center',verticalalignment='top')

    # plot encoder activity
    f_encoder_state,ax = plt.subplots(n_r,n_c,figsize=figsize,sharex=True,sharey=True)
    if len(ax.shape) < 2:
        ax = ax[:,None]
    n_p = enc.shape[-1] // trg.shape[-1]
    for n in range(n_r*n_c):
        r_idx = n // n_c
        c_idx = n % n_c
        p_idx = n*n_p + np.arange(n_p)
        if n < n_ch:
            ax[r_idx,c_idx].plot(plot_t_in,enc[:,p_idx],color=c_encoder,alpha=1/np.sqrt(n_p)) # alpha's a weird heuristic here
            ax[r_idx,c_idx].get_xaxis().set_ticks([])
            ax[r_idx,c_idx].get_yaxis().set_ticks([])
            # ax[r_idx,c_idx].legend('{}'.format(n))
        plt.sca(ax[r_idx,c_idx])
        plt.box(on=False)
        plt.xticks([])
        plt.yticks([])
    # add time, amplitude references
    x_min, x_max, y_min, y_max = ax[r_idx,0].axis()
    x_w_in = bar_t_in/(x_max-x_min)
    y_w = bar_a_enc/(y_max-y_min)
    # amplitude scale bar
    ax[r_idx,0].axvline(x_min,ymin=0,ymax=y_w,linewidth=3,color='k')
    ax[r_idx,0].text(x_min,y_min+bar_a_enc/2,"{} ".format(bar_a_enc),
                     horizontalalignment='right',verticalalignment='center')
    # time scale bar
    ax[r_idx,0].axhline(y_min,xmin=0,xmax=x_w_in,linewidth=3,color='k')
    ax[r_idx,0].text(x_min+bar_t_in/2,y_min,"{} s".format(bar_t_in),
                     horizontalalignment='center',verticalalignment='top')

    # plot decoder activity
    f_decoder_state,ax = plt.subplots(n_r,n_c,figsize=figsize,sharex=True,sharey=True)
    if len(ax.shape) < 2:
        ax = ax[:,None]
    n_p = dec.shape[-1] // trg.shape[-1]
    for n in range(n_r*n_c):
        r_idx = n // n_c
        c_idx = n % n_c
        p_idx = n*n_p + np.arange(n_p)
        if n < n_ch:
            ax[r_idx,c_idx].plot(plot_t_out,dec[:,p_idx],color=c_decoder,alpha=1/np.sqrt(n_p))
            ax[r_idx,c_idx].get_xaxis().set_ticks([])
            ax[r_idx,c_idx].get_yaxis().set_ticks([])
            # ax[r_idx,c_idx].legend('{}'.format(n))
        plt.sca(ax[r_idx,c_idx])
        plt.box(on=False)
        plt.xticks([])
        plt.yticks([])
    # add time, amplitude references
    x_min, x_max, y_min, y_max = ax[r_idx,0].axis()
    x_w_in = bar_t_out/(x_max-x_min)
    y_w = a_dec/(y_max-y_min)
    # amplitude scale bar
    ax[r_idx,0].axvline(x_min,ymin=0,ymax=y_w,linewidth=3,color='k')
    ax[r_idx,0].text(x_min,y_min+a_dec/2,"{}".format(bar_a_dec),
                     horizontalalignment='right',verticalalignment='center')
    # time scale bar
    ax[r_idx,0].axhline(y_min,xmin=0,xmax=x_w_in,linewidth=3,color='k')
    ax[r_idx,0].text(x_min+bar_t_out/2,y_min,"{} s".format(bar_t_out),
                     horizontalalignment='center',verticalalignment='top')

    return f_target_v_out, f_encoder_state, f_decoder_state, f_source

# silly tool to format epoch computation times
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class ECOGLoss(_Loss):
    # __constants__ = ['objective']
    def __init__(self,objective='L1',sum_axis=(-1,-2)):
        super(_Loss, self).__init__()
        self.objective = objective
        self.sum_axis = sum_axis

    def forward(self,input,target):
        # compute individual loss from all channels, all batch elements
        if self.objective in ['L1','l1','LASSO','lasso','Lasso','abs']:
            loss_unreduced = l1_loss(input,target,reduction='none')
        if self.objective in ['L2','l2','MSE']:
            loss_unreduced = mse_loss(input,target,reduction='none')
        loss = loss_unreduced.sum(axis=self.sum_axis).mean()
        return loss
