import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

class LSV1M_Dataset(Dataset):
    def __init__(self, resps, stims, neuron_idxs=None, names=("inputs", "targets")):
        super().__init__()
        
        self.stims = torch.Tensor(stims).unsqueeze(1)
        self.resps = torch.Tensor(resps)
        self.DataPoint = namedtuple("DataPoint", names)

    def __len__(self):
        return len(self.stims)
    
    def __getitem__(self, index):
        tensors_expanded = [self.stims[index], self.resps[index]]
        return self.DataPoint(*tensors_expanded)


def get_LSV1M_dataloaders(
    population='both', 
    n_images_val=5000, 
    exc_neuron_idxs=np.arange(0,37500),
    inh_neuron_idxs=np.arange(0,9375), 
    normalize_input=True, 
    center_input=True,
    normalize_target=True,
    batch_size=16):

    if population == 'both':
        neuron_idxs = list(exc_neuron_idxs) + list(inh_neuron_idxs + 37500)
        resps = [
            '/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy',
            '/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy'
            ]
    if population == 'exc':
        neuron_idxs = list(exc_neuron_idxs) 
        resps = ['/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy']
    if population =='inh':
        neuron_idxs = list(inh_neuron_idxs)
        resps = ['/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy']

    print('loading data')
    resps = np.concatenate([np.load(resp) for resp in resps], -1)
    stims = np.load('/CSNG/baroni/Dic23data/all_single_trial_stims55x55.npy')

    print('preprocessing data')
    if normalize_input==True:
        stims_mean = stims.mean()
        stims_std = stims.std()
        stims = (stims - stims_mean)/stims_std

    if center_input==True:
        stims = stims - (stims.max() + stims.min())/2

    if normalize_target==True:
        resps_mean = resps.mean(axis=0, keepdims=True)
        resps_std =  resps.std(axis=0, keepdims=True)
        resps = (resps - resps_mean)/resps_std

    ds_train =  LSV1M_Dataset(resps = resps[:-n_images_val, neuron_idxs], stims = stims[:-n_images_val])
    ds_validation = LSV1M_Dataset(resps = resps[-n_images_val:, neuron_idxs], stims = stims[-n_images_val:])

    dataloaders = {
        'train': {'all_sessions': DataLoader(ds_train, batch_size=batch_size, shuffle=True)},
        'validation': {'all_sessions': DataLoader(ds_validation, batch_size=batch_size, shuffle=False)}
    }

    print('loading test data')
    resps_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_resps.npy')
    stims_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55.npy')
    
    # preprocess stims
    print('preprocessing test data')
    if normalize_input==True:
        stims_multi = (stims_multi - stims_mean)/stims_std
    if center_input==True:
        stims_multi = stims_multi - (stims_multi.max() + stims_multi.min())/2
    if normalize_target==True:
        resps_multi = (resps_multi - resps_mean.reshape(1,1,-1))/resps_std.reshape(1,1,-1)
    if population =='inh':
        resps_multi = resps_multi[:, :, list(np.array(neuron_idxs) + 37500)]
    else:
        resps_multi = resps_multi[:, :, neuron_idxs]
        
    ds_test =  LSV1M_Dataset(resps = resps_multi.mean(axis=1), stims = stims_multi)
    dataloaders['test'] =  {'all_sessions': DataLoader(ds_test, batch_size=16, shuffle=True, num_workers=0)}
    return dataloaders