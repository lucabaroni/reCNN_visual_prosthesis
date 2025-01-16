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
from LSV1M_training.dataset import get_LSV1M_dataloaders


seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


lr_init = 0.001
batch_size = 300
upsampling= 1
rot_eq_batch_norm = True
hidden_channels = 1
hidden_kern = 5
input_kern = 3
layers = 5
gamma_hidden = 0.02
gamma_input = 0.02
depth_separable = True
init_sigma_range = 0.02
fine_tune = 'core'
population = 'both'
adamW= False
n_exc = 20000
n_inh = 5000
lr_decay_steps=4
patience=5

exc_neuron_idxs = np.arange(0,n_exc)
inh_neuron_idxs = np.arange(0,n_inh)


dataloaders = get_LSV1M_dataloaders(
    population=population, 
    batch_size=batch_size,
    n_images_val=5000, 
    exc_neuron_idxs=exc_neuron_idxs,
    inh_neuron_idxs=inh_neuron_idxs)

# trainer
trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 1,
    'lr_init':  lr_init,
    'lr_decay_steps': lr_decay_steps,
    'patience': lr_decay_steps,
    'track_training':False,
    'verbose': True,
    'adamw': adamW, 
    'fine_tune': fine_tune})

trainer = get_trainer(trainer_fn, trainer_config)

model_config = dict(
    num_rotations=  32, 
    stride=1, 
    input_regularizer='LaplaceL2norm',
    upsampling = upsampling,
    rot_eq_batch_norm = rot_eq_batch_norm,
    hidden_channels = hidden_channels,
    hidden_kern = hidden_kern,
    input_kern = input_kern,
    layers = layers,
    gamma_hidden = gamma_hidden,
    gamma_input = gamma_input,
    depth_separable = depth_separable,
    init_sigma_range= init_sigma_range,
    use_avg_reg=False, 
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda()

#load pos and ori 
neuron_idxs = np.concatenate([exc_neuron_idxs, inh_neuron_idxs+37500])
pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']]))/(5.5)
pos2 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']]))/(5.5)
ori = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']])/np.pi)
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1[neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = -pos2[neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori[neuron_idxs]

val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)

test_corr =  get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=True)
results = {
    'val/corr' : output['validation_corr'], 
    # 'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    # 'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': test_corr.mean(), 
    'test/corr_per_neuron': test_corr, 
    # 'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
}


#%%
model_state_dict = model.state_dict()
core_sd = {}
for k, v in model.state_dict().items():
    if k.startswith('core'):
        core_sd[k] = v
#%%
left_out_exc_neurons = np.arange(n_exc, 37500)
left_out_inh_neurons = np.arange(n_inh, 9375)

dataloaders_left_out = get_LSV1M_dataloaders(
    population=population, 
    batch_size=batch_size,
    n_images_val=5000, 
    exc_neuron_idxs=left_out_exc_neurons,
    inh_neuron_idxs=left_out_inh_neurons)
#%%
model = BRCNN_no_scaling(dataloaders_left_out, seed=0, **model_config)
model.load_state_dict(core_sd, strict=False)
model.cuda()

left_out_neuron_idxs = np.concatenate([left_out_exc_neurons, left_out_inh_neurons+37500])
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1[left_out_neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = -pos2[left_out_neuron_idxs]
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori[left_out_neuron_idxs]

test_corr =  get_correlations(model, dataloaders_left_out['test'], as_dict=False, per_neuron=True)
results['test/avg_corr_left_out'] = test_corr.mean()


