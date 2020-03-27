from torch.utils.data import Dataset, DataLoader

# dataset interface to ECoG data (really more general, just multivariate time series data)
# to-do: expanding this to 
class EcogDataset(Dataset):
    def __init__(self, data_in, block_len):
        self.data = data_in
        self.block_len = int(block_len)
        self.data_len = self.data.shape[0]

    def __len__(self):
        return self.data.shape[0] // self.block_len

    def __getitem__(self, idx):
        # get data range (make sure not to sample from outside range)
        return self.data[idx:(idx + self.block_len),:]

# modified batch loader for sequential ECoG (or general time series data)
class BatchEcogSampler(Sampler):
    # produces BATCH_SIZE draws from the EcogDataset interface
    # ensures that a full sequence is drawn from the dataset
    def __init__(self,Sampler,batch_size,drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        

# produce sequential dataloaders
def genLoaders( dataset, train_frac=0.8, test_frac=0.1, valid_frac=0.1, BATCH_SIZE=5):
    data_size = dataset.data.size[-1]
    seq_len = data.set.block_len
    idx_all = np.arange(data_size)
    train_split = int(np.floor(train_frac*data_size))
    test_split = int(np.floor(test_frac*data_size))
    valid_split = int(np.floor(valid_frac*data_size))
    
    train_idx = idx_all[0:train_split:seq_len]
    test_idx  = idx_all[train_split+valid_split:-1:seq_len]
    valid_idx = idx_all[train_split:train_split+valid_split:seq_len]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler,
                                               drop_last=True) # this can be avoided using some padding sequence classes, I think
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                              sampler=test_sampler,
                                              drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=valid_sampler,
                                               drop_last=True)
    
    return train_loader, valid_loader, test_loader