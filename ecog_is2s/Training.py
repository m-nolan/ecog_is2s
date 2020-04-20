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
#     widgets = [pb.Percentage(), progressbar.Bar()]
#     bar = pb.ProgressBar(widgets=widgets).start()
    for i, batch in enumerate(iterator):
        if np.mod(i+1,1000) == 0:
            print(i,len(iterator))
        src = batch[:,:-1,:]
        trg = batch[:,-1,:].unsqueeze(1) # otherwise it would automatically cut this out.

        optimizer.zero_grad()

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
    
    with torch.no_grad():
#         widgets = [pb.Percentage(), progressbar.Bar()]
#         bar = pb.ProgressBar(widgets=widgets).start()
#         i = 0
        for i, batch in enumerate(iterator):

            if np.mod(i+1,1000)==0:
                print(i,len(iterator))
            src = batch[:,:-1,:]
            trg = batch[:,-1,:].unsqueeze(1)

            output = model(src, trg, 0.) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            batch_loss.append(loss.item())
            
        if plot_flag:
            plot_output = (src,trg,output)
        else:
            plot_output = ()
        
    return epoch_loss / len(iterator), np.array(batch_loss), plot_output

def eval_plot(plot_dict,figsize=(10,8),n_pca=1):
    # compute PCA dims from catted src/trg data
    # make sure to push this frame back to the cpu!
    src = plot_dict['src'].cpu().squeeze(dim=0)
    trg = plot_dict['trg'].cpu().squeeze(dim=0)
    out = plot_dict['out'].cpu().squeeze(dim=0)
#     print(plot_dict['src'].shape,plot_dict['trg'].shape)
    full_train = np.vstack((src,trg))
    full_train_n = full_train.shape[0]
    full_train_mean = np.mean(full_train,axis=0)
    full_train_cov = np.matmul((full_train-full_train_mean).T,(full_train-full_train_mean))
    w, v = np.linalg.eig(full_train_cov) # w is the eval, v is the evec (columns)
    target_red = np.matmul(trg,v[:,:n_pca])
    output_red = np.matmul(out,v[:,:n_pca])
    target_n = target_red.shape[0]
    plot_t = np.arange(target_n)/plot_dict['srate']
    print(plot_t)
    f,ax = plt.subplots(n_pca,1,figsize=(figsize[0],n_pca*figsize[1]))
    if n_pca == 1:
        ax = [ax]
    for n in range(n_pca):
        ax[n].plot(plot_t,target_red[:,n],label='trg_{}'.format(n))
        print(target_red[:,n])
        ax[n].plot(plot_t,output_red[:,n],label='out_{}'.format(n))
        print(output_red[:,n])
        ax[n].legend(loc=0)
        ax[n].set_xlabel('time (s)')
        ax[n].set_ylabel('PC{}'.format(n))
    return f, ax

# silly tool to format epoch computation times
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# def plot_example_sequence(data_tuple,figsize=())