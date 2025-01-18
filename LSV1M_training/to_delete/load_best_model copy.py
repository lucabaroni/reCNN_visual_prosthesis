#%%
from unittest import result
import classical_exps.functions.experiments 
import torch
import wandb
import yaml
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 
from nnfabrik.builder import get_model, get_trainer, get_data
import nnvision
from models import EnergyModel
from pickle_utils import pickleread
# from utils import get_config
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE
from models import BRCNN_with_common_scaling, BRCNN_no_scaling, single_BRCNN_no_scaling
from torch.utils.data import Dataset, DataLoader
import os 
from collections import namedtuple
import wandb

x = torch.load('/project/check_trained_model/artifacts/core_sd:v3/core_sd.pth')

x.keys()
# %% LOAD THE BEST MODEL

api = wandb.Api()

# Access the run
run = api.run('lucabaroni/sweep_LSV1M/z2xy46u2')

# Print the run's summary (a general overview)
print("Run Summary:", run.summary)

# Print the configuration used in the run
print("Run Config:", run.config)

# Print system metrics
print("System Metrics:", run.system_metrics)

# Get all artifacts associated with this run
artifacts = run.logged_artifacts()
for artifact in artifacts:
    print(f"Artifact: {artifact.name}")
    print(f"Artifact Metadata: {artifact.metadata}")


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

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


batch_size = 300
population = 'both'

n_images_val = 5000
exc_neuron_idxs = np.arange(20000)
inh_neuron_idxs = np.arange(5000)

if population == 'both':
    neuron_idxs = list(exc_neuron_idxs) + list(inh_neuron_idxs + 37500)
if population == 'exc':
    neuron_idxs = list(exc_neuron_idxs) 
if population =='inh':
    neuron_idxs = list(inh_neuron_idxs + 37500)

#%% load data
print('loading data')
resps = [
    '/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy',
    '/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy'
]
resps = np.concatenate([np.load(resp) for resp in resps], -1)
stims = np.load('/CSNG/baroni/Dic23data/all_single_trial_stims55x55.npy')

print('preprocessing data')
stims_mean = stims.mean()
stims_std = stims.std()
stims = (stims - stims_mean)/stims_std
resps_mean = resps.mean(axis=0, keepdims=True)
resps_std =  resps.std(axis=0, keepdims=True)
resps = (resps - resps_mean)/resps_std

#%% 
stims = np.zeros([100000, 55, 55])
resps = np.zeros([100000, 46875])

ds_train =  LSV1M_Dataset(resps = resps[:-n_images_val, neuron_idxs], stims = stims[:-n_images_val])
ds_validation = LSV1M_Dataset(resps = resps[-n_images_val:, neuron_idxs], stims = stims[-n_images_val:])

dataloaders = {
    'train': {'all_sessions': DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)},
    'validation': {'all_sessions': DataLoader(ds_validation, batch_size=batch_size, shuffle=False, num_workers=0)}
}
config = run.config

# Set hyperparameters
lr_init = config['lr_init']
batch_size = config['batch_size']
upsampling = config['upsampling'] if 'upsampling' in config else 1
rot_eq_batch_norm = config['rot_eq_batch_norm'] if 'rot_eq_batch_norm' in config else True
hidden_channels = config['hidden_channels']
hidden_kern = config['hidden_kern']
input_kern = config['input_kern'] if 'input_kern' in config else 3
layers = config['layers']
gamma_hidden = 0 if config['gamma_hidden'] == 0 else config['gamma_hidden_nonzero']
gamma_input = 0 if config['gamma_input'] == 0 else config['gamma_input_nonzero']
depth_separable = config['depth_separable']
init_sigma_range = config['init_sigma_range']
fine_tune = config['fine_tune'] if 'fine_tune' in config else 'core'
population = config['population'] if 'population' in config else 'both'
adamW = config['adamW'] if 'adamW' in config else False
n_exc = config['n_exc'] if 'n_exc' in config else 20000
n_inh = config['n_inh'] if 'n_inh' in config else 5000
lr_decay_steps = config['lr_decay_steps'] if 'lr_decay_steps' in config else 4
patience = config['patience'] if 'patience' in config else 5

# Model configuration
model_config = dict(
    num_rotations=32,
    stride=1,
    input_regularizer='LaplaceL2norm',
    upsampling=upsampling,
    rot_eq_batch_norm=rot_eq_batch_norm,
    hidden_channels=hidden_channels,
    hidden_kern=hidden_kern,
    input_kern=input_kern,
    layers=layers,
    gamma_hidden=gamma_hidden,
    gamma_input=gamma_input,
    depth_separable=depth_separable,
    init_sigma_range=init_sigma_range,
    use_avg_reg=False,
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda()
model.load_state_dict(x, strict=False)

neuron_idxs = np.concatenate([exc_neuron_idxs, inh_neuron_idxs + 37500])
pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']])) / (5.5)
pos2 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']])) / (5.5)
ori = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']]) / np.pi)
model.readout['all_sessions'].mu.data[0, 0, :, 0, 0] = pos1[neuron_idxs]
model.readout['all_sessions'].mu.data[0, 0, :, 0, 1] = -pos2[neuron_idxs]
model.readout['all_sessions'].mu.data[0, 0, :, 0, 2] = ori[neuron_idxs]

#%% test
resps_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_resps.npy')
stims_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55.npy')

print('preprocessing')
stims_multi = (stims_multi - stims_mean)/stims_std
resps_multi = (resps_multi - resps_mean)/resps_std
ds_test =  LSV1M_Dataset(resps = resps_multi.mean(axis=1)[:, neuron_idxs], stims = stims_multi)
dataloaders['test'] =  {'all_sessions': DataLoader(ds_test, batch_size=16, shuffle=True, num_workers=0)}

results = {
    'val/corr' : get_correlations(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    # 'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    # 'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
    # 'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
}
# %%

# %%
