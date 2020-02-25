import torch.nn as nn

# initialize model weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# return an total count of model parameters.
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# get epoch time stats, what a cutie
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
