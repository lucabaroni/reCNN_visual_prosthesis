#%%
from nnfabrik.builder import get_model, get_trainer, get_data
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 
import nnvision
from models import EnergyModel
from pickle_utils import pickleread
from utils import get_config
from nnvision.datasets.monkey_loaders import monkey_static_loader_combined_modified
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict

import wandb

dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]

import numpy as np

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_MSE',
    'loss_function': 'MSE',
    'maximize': False,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 200,
    'lr_init': 0.5,
    # lr_init': 0.00005,
    'lr_decay_steps': 4,
    'patience': 5,
    'track_training':True,
    'verbose': True,
    'adamw': True})

trainer = get_trainer(trainer_fn, trainer_config)


dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
    {'dataset': 'CSRF19_V1',
    'neuronal_data_files': ['/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'],
    'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
    'crop': 70,
    'subsample': 1,
    'seed': 1000,
    'scale': 0.5,
    'time_bins_sum': 12,
    'batch_size': 128, 
    'normalize_resps':True})

dataloaders = get_data(dataset_fn, dataset_config)
data = pickleread('/project/data_all_mei_and_ori.pickle')
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
    pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
    ori.append(data[i]['preferred_ori']/180)
pos1 = torch.Tensor(pos1)
pos2 = torch.Tensor(pos2)
ori = torch.Tensor(ori)

config={}

model_config = dict(
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    hidden_channels = 16, 
    hidden_kern = 5, 
    input_kern = 7,
    # layers = 3, 
    gamma_hidden = 0.02,
    gamma_input =  0.2,
    # depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=True, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,
)
from models import BRCNN_with_common_scaling

model = BRCNN_with_common_scaling(dataloaders, seed=0, **model_config)

model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori

# todo add initialization at right spot

#%%
# # Custom function to enforce sign constraints
# def enforce_sign_constraint(param):
#     with torch.no_grad():
#         param[param < 0.02] = 0.02

# # Register the hook directly to the final_scale parameter
# if hasattr(model, 'final_scale'):
#     model.final_scale.register_hook(lambda grad: enforce_sign_constraint(model.final_scale))

trainer(model, dataloaders, seed=seed)
# %%

x = torch.randn(1,1,46,46).cuda()
# %%


