#%%
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
from utils import get_config
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE
from models import BRCNN_with_common_scaling, BRCNN_no_scaling, single_BRCNN_no_scaling
from torch.utils.data import Dataset, DataLoader
import os 
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

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

lr_init = 0.001
batch_size = 300
upsampling= 1
rot_eq_batch_norm = True
hidden_channels = 8
hidden_kern = 7
input_kern = 3
layers = 5
gamma_hidden = 0.02
gamma_input = 0.02
depth_separable = True
init_sigma_range = 0.02
fine_tune = 'core'
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

# trainer
trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 1,
    'lr_init':  lr_init,
    'lr_decay_steps': 1,
    'patience': 1,
    'track_training':False,
    'verbose': True,
    'adamw': True, 
    'fine_tune': fine_tune})

trainer = get_trainer(trainer_fn, trainer_config)

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
ds_train =  LSV1M_Dataset(resps = resps[:-n_images_val, neuron_idxs], stims = stims[:-n_images_val])
ds_validation = LSV1M_Dataset(resps = resps[-n_images_val:, neuron_idxs], stims = stims[-n_images_val:])

dataloaders = {
    'train': {'all_sessions': DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)},
    'validation': {'all_sessions': DataLoader(ds_validation, batch_size=batch_size, shuffle=False, num_workers=0)}
}

#%%
model_config = dict(
    num_rotations=  32, 
    stride=1, 
    input_regularizer='LaplaceL2norm',
    upsampling = 1,
    rot_eq_batch_norm = True,
    hidden_channels = 8,
    hidden_kern = 5,
    input_kern = 3,
    layers = 5,
    gamma_hidden = 0,
    gamma_input = 0,
    depth_separable = True,
    init_sigma_range= 0.2,
    use_avg_reg=False, 
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda()

#load pos and ori 
pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']]))/(5.5)
pos2 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']]))/(5.5)
ori = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']])/np.pi)
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1[neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = -pos2[neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori[neuron_idxs]

#%% train 
val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)

#%% test
resps_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_resps.npy')
stims_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55.npy')

# preprocess stims
print('preprocessing')
stims_multi = (stims_multi - stims_mean)/stims_std
resps_multi = (resps_multi - resps_mean)/resps_std

#%% average test set
ds_test =  LSV1M_Dataset(resps = resps_multi.mean(axis=1)[:, neuron_idxs], stims = stims_multi)
dataloaders['test'] =  {'all_sessions': DataLoader(ds_test, batch_size=16, shuffle=True, num_workers=0)}

#%%
results = {
    'val/corr' : output['validation_corr'], 
    # 'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    # 'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
    # 'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
}

# wandb.log(results)
# wandb.run.summary.update(results)

best_val_corr = torch.load('/project/monkey_training/best_val_corr.pt')

if  output['validation_corr'] > best_val_corr:
    torch.save(output['validation_corr'], '/project/monkey_training/best_val_corr.pt')
    torch.save(
        {
            'model_config': model_config, 
            'model_state_dict': model.state_dict()
        },
        '/project/monkey_training/best_model.pt')

#%%
# %%
from tqdm import tqdm

up_to_50k_stims = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
from50k_stims = np.load('/CSNG/baroni/Dic23data/all_single_trial_stims.npy')
stims100k = np.concatenate([up_to_50k_stims, from50k_stims])
np.save(stims100k, '/CSNG/baroni/Dic23data/100k_single_trial_stims.npy')
# %%
up_to_50k_Exc = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
from50k_Exc = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy')
Exc100k = np.concatenate([up_to_50k_Exc, from50k_Exc])
np.save(Exc100k, '/CSNG/baroni/Dic23data/100k_single_trial_V1_Exc_L23.npy')
#%%
up_to_50k_Inh = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
from50k_Inh = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy')
Inh100k = np.concatenate([up_to_50k_Inh, from50k_Inh])
np.save(Inh100k, '/CSNG/baroni/Dic23data/100k_single_trial_V1_Inh_L23.npy')