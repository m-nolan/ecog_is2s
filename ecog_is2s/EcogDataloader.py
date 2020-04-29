from torch.utils.data import Dataset, DataLoader
from torch import randperm
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np


# create device mounting function (move data to GPU)
def to_device( data, device ):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# dataset interface to ECoG data (really more general, just multivariate time series data)
class EcogDataset(Dataset):
    def __init__(self, data_in, device, block_len):
        self.data = data_in
        self.device = device
        self.block_len = int(block_len)
        self.data_len = self.data.shape[0]

    def __len__(self):
        return self.data.shape[0] // self.block_len #?

    def __getitem__(self, idx):
        # get data range (make sure not to sample from outside range)
        return self.data[idx:(idx + self.block_len),:].to(self.device, non_blocking=True)

# produce sequential dataloaders
def genLoaders( dataset, sample_idx, train_frac, test_frac, valid_frac, batch_size, rand_samp=False, plot_seed=0):
    data_size = dataset.data.shape[0]
    idx_all = np.arange(data_size)
#     smpl_idx_all = idx_all[:-seq_len:seq_len]

    train_sampler, test_sampler, valid_sampler, plot_sampler = genSamplers(
        sample_idx,train_frac,test_frac,valid_frac=valid_frac,rand_samp=rand_samp,plot_seed=plot_seed
        )

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              drop_last=True) # this can be avoided using some padding sequence classes, I think
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             drop_last=True)
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              drop_last=True)
    plot_loader = DataLoader(dataset,
                             batch_size=1,
                             sampler=plot_sampler,
                             drop_last=False)
    return train_loader, test_loader, valid_loader, plot_loader


def genSamplers( idx, train_frac, test_frac, valid_frac=0.0, rand_samp=False, plot_seed=0, verbose=True ):
    n_samp = len(idx)
    # this allows you to randomly shuffle the train/test splits.
    # Turns out, you don't want this.
    if rand_samp:
        shuffle_idx = randperm(n_samp)
    else: # this gives a straight p_tr, p_te, p_va split of the data. Will add more flexible breaks later. \////\
        suffle_idx = np.arange(n_samp)
    train_split = int(np.floor(train_frac*n_samp))
    valid_split = int(np.floor(valid_frac*n_samp)) # save this for a more
    test_split = int(np.floor(test_frac*n_samp))
    # rework this using
    train_idx = idx[shuffle_idx[:train_split]]
    valid_idx = idx[shuffle_idx[train_split:train_split+valid_split]]
    test_idx = idx[shuffle_idx[train_split+valid_split:-1]]
    plot_idx = np.array([train_idx[plot_seed], test_idx[plot_seed], valid_idx[plot_seed]])
    if verbose:
        print(train_idx.shape,test_idx.shape,valid_idx.shape,plot_idx.shape)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    plot_sampler = SubsetRandomSampler(plot_idx)

    return train_sampler, test_sampler, valid_sampler, plot_sampler
