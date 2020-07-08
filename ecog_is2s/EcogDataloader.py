from torch.utils.data import Dataset, DataLoader, Sampler
from torch import randperm
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import numpy as np


# create device mounting function (move data to GPU)
def to_device( data, device ):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# dataset interface to ECoG data (really more general, just multivariate time series data)
class EcogDataset(Dataset):
    def __init__(self, data_in, device, block_len):
        data_in_dimensions = data_in.shape
        # check signal size, pad with empty channel dimension if necessary.
        if len(data_in_dimensions) < 2:
            data_in = data_in[:,None]
            raise Warning("Input dimension detected: {}. Padding to fill 2d-array requirement.")
        if len(data_in_dimensions) > 2:
            raise Exception("Input dimension detected: {}. Input data array must be 2-dimensional.".format(data_in_dimensions))
        self.data = data_in
        self.device = device
        self.block_len = int(block_len)
        # self.return_diff = return_diff # idea - implement dX computation at loading point?
        self.data_len = self.data.shape[0]
        self.n_ch = self.data.shape[-1]

    def __len__(self):
        return self.data.shape[0] // self.block_len #?

    def __getitem__(self, idx):
        # get data range (make sure not to sample from outside range)
        data_out = self.data[idx:(idx + self.block_len),:].to(self.device, non_blocking=True)
        return data_out

# produce sequential dataloaders
def genLoaders( dataset, sample_idx, train_frac, test_frac, valid_frac, batch_size, drop_last=False, rand_samp=False, plot_seed=0):
    data_size = dataset.data.shape[0]
    idx_all = np.arange(data_size)
#     smpl_idx_all = idx_all[:-seq_len:seq_len]

    train_sampler, test_sampler, valid_sampler, plot_sampler = genSamplers(
        sample_idx,train_frac,test_frac,valid_frac=valid_frac,rand_samp=rand_samp,plot_seed=plot_seed
        )

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              drop_last=drop_last) # this can be avoided using some padding sequence classes, I think
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             drop_last=drop_last)
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              drop_last=drop_last)
    plot_loader = DataLoader(dataset,
                             batch_size=1,
                             sampler=plot_sampler,
                             drop_last=drop_last)
    return train_loader, test_loader, valid_loader, plot_loader


def genSamplers( idx, train_frac, test_frac, valid_frac=0.0, rand_samp=False, plot_seed=0, verbose=False ):
    n_samp = len(idx)
    # this allows you to randomly shuffle the train/test splits.
    # Turns out, you don't want this.
    if rand_samp:
        shuffle_idx = randperm(n_samp)
    else: # this gives a straight p_tr, p_te, p_va split of the data. Will add more flexible breaks later. \////\
        shuffle_idx = np.arange(n_samp)
    train_split = int(np.floor(train_frac*n_samp))
    valid_split = int(np.floor(valid_frac*n_samp))
    test_split = int(np.floor(test_frac*n_samp))
    train_idx = idx[shuffle_idx[:train_split]]
    valid_idx = idx[shuffle_idx[train_split:train_split+valid_split]]
    test_idx = idx[shuffle_idx[train_split+valid_split:-1]]
    plot_idx = np.concatenate((train_idx[plot_seed], test_idx[plot_seed]))#, valid_idx[plot_seed]])
    if verbose:
        print(train_idx.shape,test_idx.shape,valid_idx.shape,plot_idx.shape)

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    plot_sampler = SequentialArraySampler(plot_idx)

    return train_sampler, test_sampler, valid_sampler, plot_sampler

class SequentialArraySampler(Sampler):
    r"""Samples elements sequentially from the given input listlike data_source.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)
