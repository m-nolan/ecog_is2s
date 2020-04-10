import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math
import time

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

def evaluate(model, iterator, criterion):
    
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

#             output = output[1:].view(-1, output_dim)
#             trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)

            epoch_loss += loss.item()
            batch_loss.append(loss.item())

#             bar.update(i/10000)

#             if i > 10000:
#                 break
#             i += 1
        
    return epoch_loss / len(iterator), np.array(batch_loss)

# silly tool to format epoch computation times
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# # the actual training loop!
# N_EPOCHS = 10 # hahahaha
# CLIP = 1

# best_valid_loss = float('inf')
# for epoch in range(N_EPOCHS):
#     start_time = time.time()
    
#     # the data splitter makes the training, validation and test sets.
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion)
    
#     end_time = time.time()
    
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'ex1-model.pt')
    
#     print(f'Epoch: {epoch+1:02} | time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# model.load_state_dict(torch.load('tut1-model.pt'))
# test_loss = evaluate(model, test_iterator, criterion)

# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
