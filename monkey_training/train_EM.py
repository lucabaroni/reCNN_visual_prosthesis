#%%
from nnfabrik.builder import get_model, get_trainer, get_data
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 
import nnvision
from pickle_utils import pickleread
from utils import get_config
from nnvision.datasets.monkey_loaders import monkey_static_loader_combined

# %load_ext autoreload

# %autoreload 2

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
{
'stop_function': 'get_MSE',
'loss_function': 'MSE',
'maximize': False,
'avg_loss': False,
'device': 'cuda',
'max_iter': 200,
'lr_init': 0.005,
# lr_init': 0.00005,
'lr_decay_steps': 2,
'patience': 2,
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
    'batch_size': 16, 
    'normalize_resps': False})

dataloaders = get_data(dataset_fn, dataset_config)

data = pickleread('/project/data_all_mei_and_ori.pickle')
pos1 = []
pos2 = []
ori = []

for i in range(458):
    pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
    pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
    ori.append((-data[i]['preferred_ori']*np.pi/180)) 

pos1 = torch.Tensor(pos1)
pos2 = torch.Tensor(pos2)
ori = torch.Tensor(ori)
# create (or load) energy model

config=dict(
    positions_x = pos1,
    positions_y = pos2,
    orientations = ori,
    sigma_x_init = 0.2,
    sigma_y_init= 0.2,
    f_init = 1, 
    filter_scale_init = 0.01, 
    final_scale_init = 1,
    resolution = (46,46),
    xlim = (-2.67/2, 2.67/2),
    ylim = (-2.67/2, 2.67/2), 
    nonlinearity='square_root', 
    common_rescaling=False,

)
model_fn, model_config = ('models.EnergyModel', config)

model = get_model(model_fn, model_config, dataloaders=dataloaders, seed = 42)

#%%
trainer(model, dataloaders, seed=seed)


#%%

from nnvision.utility.measures import get_correlations, get_avg_correlations
from nnfabrik.builder import get_model, get_trainer, get_data
# from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext, v1_convnext_ensemble, color_v1_convnext_ensemble

avg_corr = get_avg_correlations(model, dataloaders, 'cuda')
print(avg_corr.mean())

model.final_bias.data = torch.zeros_like(model.final_bias.data)
model.final_scale.data = torch.ones_like(model.final_scale.data)

avg_corr = get_avg_correlations(model, dataloaders, 'cuda')


plt.hist(avg_corr)
plt.show()

# torch.save(model.state_dict(), f"encoder_3nd_test_longer_no_bias2.pt")

# model.cuda().train()
# model.readout['all_sessions'].mu[0,0,:,0,0] = pos1
# model.readout['all_sessions'].mu[0,0,:,0,1]  = pos2
# model.readout['all_sessions'].mu[0,0,:,0,2]  = ori

# import matplotlib.pyplot as plt

# n = 6
# plt.imshow(data[n]['mei'])
# plt.show()
# plt.imshow(model.get_filters_even_and_odd()[0].squeeze().detach().cpu().numpy()[n])
# plt.show()

#%%
# NORMALIZE RESPS
# resp = torch.cat([x[1] for x in dataloaders['train']['all_sessions']], 0).T
# shown = torch.cat([x[2] for x in dataloaders['train']['all_sessions']], 0).T
# means = torch.Tensor([resp[i][shown[i]==True].mean() for i in range(458)])
# stds = torch.Tensor([resp[i][shown[i]==True].std() for i in range(458)])
# resps_new = []
# bools_new = []
# with torch.no_grad():
#     for datapoint in dataloaders['train']['all_sessions'].dataset:
#         resps_new.append(datapoint.targets/stds)
#         bools_new.append(datapoint.bools)
    
# # %%
# resps_new = torch.stack(resps_new)
# bools_new = torch.stack(bools_new)
# #%%

# means = torch.Tensor([(resps_new.T[i][bools_new.T[i]==True]).mean() for i in range(458)])
# stds = torch.Tensor([(resps_new.T[i][bools_new.T[i]==True]).std() for i in range(458)])
#%%



# %%
# %%

dataloaders = get_data(dataset_fn, dataset_config)
dataset=dataloaders['train']['all_sessions'].dataset

for datapoint in dataset:
    datapoint.targets = datapoint.targets/stds

resps = torch.stack([datapoint.targets for datapoint in dataset], -1)
bools = torch.stack([datapoint.bools for datapoint in dataset], -1)
stds = torch.stack([sn_resps[sn_bools==True].std() for sn_resps, sn_bools in zip(resps, bools)])
print(stds)
# %%
