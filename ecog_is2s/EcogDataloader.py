from torch.utils.data import Dataset, DataLoader, Sampler
from torch import randperm
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import os
import sys
import glob
import json
import pickle as pkl

import numpy as np
import torch


# create device mounting function (move data to GPU)
def to_device( data, device ):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# local z-score transform
class local_zscore(object):
    # I do not know if initialization requires more detail in this case.
    def __init__(self):
        None

    def __call__(self,src,trg):
        sample = torch.cat([src,trg],dim=0)
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        src_z = (src-mean)/std
        trg_z = (trg-mean)/std
        return src_z, trg_z

class add_signal_diff(object):
    def __init__(self,axis=-1,srate=1,device='cpu'):
        self.axis=-1
        self.srate=1
        self.device=device

    def __call__(self,src,trg):
        # compute center difference dsdt estimate
        dsrc = torch.zeros(src.shape,device=self.device)
        dsrc[1:-1,:] = (src[2:,:]-src[:-2,:])/(2*self.srate)
        dsrc[0,:] = dsrc[1,:]
        dsrc[-1,:] = dsrc[-2,:]
        src_aug = torch.cat((src,dsrc),axis=-1)
        return src_aug, trg


# multifile dataset - access src/trg sequences from multiple files in a collection with a single sampling index.
class WirelessEcogDataset_MultiFile(Dataset):
    # Wireless ECoG Dataset
    # Pesaran lab data
    # AOLab
    # Michael Nolan
    # 2020.07.08

    def __init__( self, ecog_file_list, src_len, trg_len, step_len, ch_idx = None, device='cpu', transform=None ):
        super(WirelessEcogDataset_MultiFile,self).__init__()
        data_parameter_dict = self.create_parameter_dict(ecog_file_list,src_len,trg_len,step_len,ch_idx)

        ## set parameters
        self.file_list = ecog_file_list
        self.src_len = src_len
        self.trg_len = trg_len
        self.data_parameter_dict = data_parameter_dict
        self.file_ref_idx = np.cumsum([len(x) for x in data_parameter_dict['sample_idx']])
        self.file_offset_idx = np.zeros(self.file_ref_idx.shape,dtype=int)
        self.file_offset_idx[1:] = self.file_ref_idx[:-1]
        self.device = device
        self.transform = transform

    def __len__( self ):
        return sum([len(x) for x in self.data_parameter_dict['sample_idx']])

    def __getitem__( self, idx ):
        file_idx = np.arange(len(self.file_list))[idx < self.file_ref_idx][0]
        filepath = self.file_list[file_idx]
        in_file_sample_idx = idx - self.file_offset_idx[file_idx]
        dtype = self.data_parameter_dict['data_type'][file_idx]
        n_ch = self.data_parameter_dict['n_ch'][file_idx]
        ch_idx = self.data_parameter_dict['ch_idx'][file_idx]
        count = (self.src_len + self.trg_len)*n_ch
        offset = self.data_parameter_dict['sample_idx'][file_idx][in_file_sample_idx]*n_ch*dtype().nbytes
        data_sample = torch.tensor(np.fromfile(filepath,dtype=dtype,count=count,offset=offset).reshape((n_ch,self.src_len+self.trg_len),order='F').T)
        src, trg = torch.split(data_sample[:,ch_idx],[self.src_len,self.trg_len])
        if self.transform:
            src,trg = self.transform(src,trg)

        return src.to(self.device,non_blocking=True), trg.to(self.device, non_blocking=True)

    def create_parameter_dict( self, ecog_file_list, src_len, trg_len, step_len, ch_idx=None ):
        n_file = len(ecog_file_list)
        n_samp_list = []
        n_ch_list = []
        srate_list = []
        data_type_list = []
        sample_idx_list = []
        ch_idx_list = []
        ch_label_list = []
        for file in ecog_file_list:
            _n_samp, _n_ch, _srate, _data_type, _sample_idx, _ch_idx, _ch_label = self.get_ecog_file_parameters(file,src_len,trg_len,step_len,ch_idx)
            n_samp_list.append(_n_samp)
            n_ch_list.append(_n_ch)
            srate_list.append(_srate)
            data_type_list.append(_data_type)
            sample_idx_list.append(_sample_idx)
            ch_idx_list.append(_ch_idx)
            ch_label_list.append(_ch_label)
        parameter_dict = {
            'n_samp': n_samp_list,
            'n_ch': n_ch_list,
            'srate': srate_list,
            'data_type': data_type_list,
            'sample_idx': sample_idx_list,
            'ch_idx': ch_idx_list,
            'ch_label': ch_label_list
        }
        return parameter_dict


    def get_ecog_file_parameters( self, ecog_file, src_len, trg_len, step_len, ch_idx ):
        ## parse file
        data_file = os.path.basename(ecog_file)
        data_file_kern = os.path.splitext(data_file)[0]
        rec_id, microdrive_name, rec_type = data_file_kern.split('.')
        data_path = os.path.dirname(ecog_file)
        # read experiment file
        exp_file = os.path.join(data_path,rec_id + ".experiment.json")
        with open(exp_file,'r') as f:
            exp_dict = json.load(f)
        # get microdrive parameters
        microdrive_name_list = [md['name'] for md in exp_dict['hardware']['microdrive']]
        microdrive_idx = [md_idx for md_idx, md in enumerate(microdrive_name_list) if microdrive_name == md][0]
        microdrive_dict = exp_dict['hardware']['microdrive'][microdrive_idx]
        electrode_label_list = [e['label'] for e in exp['hardware']['microdrive'][0]['electrodes']]
        n_ch = len(electrode_label_list)
        # get srate
        dsmatch = re.search('clfp_ds(\d+)*',rec_type)
        if rec_type == 'raw':
            srate = experiment['hardware']['acquisition']['samplingrate']
            data_type = np.ushort
            reshape_order = 'F'
        elif rec_type == 'lfp':
            srate = 1000
            data_type = np.float32
            reshape_order = 'F'
        elif rec_type == 'clfp':
            srate = 1000
            data_type = np.float32
            reshape_order = 'F'
        elif dsmatch:
            # downsampled data - get srate from name
            srate = int(dsmatch.group(1))
            data_type = np.float32
            reshape_order = 'C' # files created with np.tofile which forces C ordering. Sorry!
        # read mask
        ecog_mask_file = os.path.join(data_path,data_file_kern + ".mask.pkl")
        with open(ecog_mask_file,"rb") as mask_f:
            mask = pkl.load(mask_f)
        mask = mask["hf"] | mask["sat"]
        if 'ch' in mask.keys():
            ch_idx = mask['ch']
        else:
            ch_idx = np.arange(n_ch)
        ch_label_list = electrode_label_list[ch_idx]
        # get params
        n_samp = len(mask)
        # create sampling index - src_len+trg_len length segments that don't include a masked sample
        _sample_idx = np.arange(n_samp-(src_len+trg_len),step=step_len,dtype=int)
        _use_sample_idx = np.zeros((len(_sample_idx)),dtype=bool) # all false
        for sidx in range(len(_sample_idx)):
            _use_sample_idx[sidx] = ~np.any(mask[_sample_idx[sidx] + np.arange(src_len+trg_len)])
        sample_idx = _sample_idx[_use_sample_idx]

        return n_samp, n_ch, srate, data_type, sample_idx, ch_idx, ch_label_list

# default dataloader wrapper, should work with 'lfp', 'clfp', 'raw'
def spontaneous_ecog( src_len, trg_len, step_len, filter_type='clfp' ):
    platform_name = sys.platform
    if platform_name == 'darwin':
        # local machine
        data_dir_path = '/Volumes/Samsung_T5/aoLab/Data/WirelessData/Goose_Multiscale_M1/180325/'
    elif platform_name == 'linux2':
        # HYAK, baby!
        data_dir_path = '/gscratch/stf/manolan/Data/WirelessData/Goose_Multiscale_M1/180325/'
    elif platform_name == 'linux':
        # google cloud, don't fail me now
        data_file_full_path = '/home/mickey/Data/WirelessData/Goose_Multiscale_M1/180325/'
    # glop the file list
    ecog_file_list = glob.glob(os.path.join(data_dir_path,'0*/*.{}.dat').format(filter_type))
    return WirelessEcogDataset_MultiFile(ecog_file_list,src_len,trg_len,src_len+trg_len)

# dataset interface to ECoG data (really more general, just multivariate time series data)
class EcogDataset(Dataset):
    def __init__(self, data_in, device, src_len, trg_len, transform=None):
        data_in_dimensions = data_in.shape
        # check signal size, pad with empty channel dimension if necessary.
        if len(data_in_dimensions) < 2:
            data_in = data_in[:,None]
            raise Warning("Input dimension detected: {}. Padding to fill 2d-array requirement.")
        if len(data_in_dimensions) > 2:
            raise Exception("Input dimension detected: {}. Input data array must be 2-dimensional.".format(data_in_dimensions))
        self.data = data_in
        self.device = device
        self.src_len = int(src_len)
        self.trg_len = int(trg_len)
        self.block_len = self.src_len + self.trg_len
        # self.return_diff = return_diff # idea - implement dX computation at loading point?
        self.data_len = self.data.shape[0]
        self.n_ch = self.data.shape[-1]
        self.transform = transform

    def __len__(self):
        return self.data.shape[0] // self.block_len #?

    def __getitem__(self, idx):
        # get data range (make sure not to sample from outside range)
        data_out = self.data[idx:(idx + self.block_len),:].to(self.device, non_blocking=True)
        src = data_out[:self.src_len]
        trg = data_out[self.src_len:]
        if self.transform:
            src, trg = self.transform(src,trg)
        return src, trg

# produce sequential dataloaders
def genLoaders( dataset, sample_idx, train_frac, valid_frac, test_frac, batch_size, drop_last=False, rand_samp=False, plot_seed=0, transform=None):
    # data_size = dataset.data.shape[0]
    # idx_all = np.arange(data_size)
#     smpl_idx_all = idx_all[:-seq_len:seq_len]

    train_sampler, test_sampler, valid_sampler, plot_sampler = genSamplers(
        sample_idx,train_frac,valid_frac,test_frac=test_frac,rand_samp=rand_samp,plot_seed=plot_seed
        )

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              drop_last=drop_last) # this can be avoided using some padding sequence classes, I think
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              drop_last=drop_last)
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             drop_last=drop_last)
    plot_loader = DataLoader(dataset,
                             batch_size=1,
                             sampler=plot_sampler,
                             drop_last=drop_last)
    return train_loader, valid_loader, test_loader, plot_loader


def genSamplers( idx, train_frac, valid_frac, test_frac=0.0, rand_samp=False, plot_seed=0, verbose=False ):
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
    if not np.shape(plot_seed):
        plot_seed = [plot_seed]
    plot_idx = np.concatenate((train_idx[plot_seed], valid_idx[plot_seed]))#, valid_idx[plot_seed]])
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
