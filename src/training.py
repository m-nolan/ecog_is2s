import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math
import time

from Encoder import Encoder_GRU
from Decoder import Decoder_GRU
from Seq2Seq import Seq2Seq_GRU
import util # init_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # a better way?

# add a whole thing HERE where we load the ECoG data from a single file.

INPUT_DIM = len(SRC.vocab) # SRC, TRG are the same time series here.
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256 # used for sequence embedding models, a la NLP
# DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5 # dropout layer chance
DEC_DROPOUT = 0.5

enc = Encoder_GRU(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder_GRU(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.appy(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')
# ^ I am not familiar with this syntax.

# define the optimizer
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss
# here's the example code, detailing how to add trial padding
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
# criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# not sure if this belongs here
def train(model, iterator, optimizer, criterion, clip):
    model.train() # turns on dropout, batchnorm
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src,trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        # you'll have to change all of this for the MSE estimator.
        # not too much, but a little.
        output = output[1:].view(-1,output_dim) # creates a view of the last dim (no new memory)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        loss.backward() # computes the loss using the backward gradient method
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() # updates values based on the 
        
        epoch_loss += loss.item()
        
    return epoch_loss/len(iterator) # a fun way of taking a mean error value

def evaluate(model, iterator, criterion):
    model.eval() # turns off dropout, batchnorm
    epoch_loss = 0
    
    with torch.no_grad(): # no opts, no grads. This speeds things up.
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src,trg,0) # no teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1,output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output,trg)
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(iterator)
    

# the actual training loop!
N_EPOCHS = 10 # hahahaha
CLIP = 1

best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    # the data splitter makes the training, validation and test sets.
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'ex1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
