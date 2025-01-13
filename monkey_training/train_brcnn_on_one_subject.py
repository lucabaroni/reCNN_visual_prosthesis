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
from models import BRCNN_with_common_scaling
from nnvision.datasets.monkey_loaders import monkey_static_loader_combined
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE


filenames = ['/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
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
    '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle']

subject_0_filenames = []
subject_1_filenames = []
ids0 = []
ids1 = []
skip = 0

for file in filenames: 
    n = len(pickleread(file)['training_responses']) 
    if pickleread(file)['subject_id']== 31: 
        subject_0_filenames.append(file)
        ids0 = ids0 + list(np.arange(skip, skip + n))

    else:
        subject_1_filenames.append(file)
        ids1 = ids1 + list(np.arange(skip, skip + n ))
    skip += n
        

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
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 200,
    'lr_init':  0.001,
    'lr_decay_steps': 4,
    'patience': 5,
    'track_training':True,
    'verbose': True,
    'adamw': True, 
    'fine_tune': None,
    })


trainer = get_trainer(trainer_fn, trainer_config)

dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
    {'dataset': 'CSRF19_V1',
    'neuronal_data_files': subject_0_filenames,
    'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
    'crop': 70,
    'subsample': 1,
    'seed': 1000,
    'scale': 0.5,
    'time_bins_sum': 12,
    'batch_size': 8, 
    'normalize_resps':True})

dataloaders = get_data(dataset_fn, dataset_config)
next(iter(dataloaders['train']['all_sessions'])).inputs.shape

model_config = dict(
    num_rotations=  32, 
    stride=1, 
    input_regularizer='LaplaceL2norm',
    upsampling = 1,
    rot_eq_batch_norm = True,
    hidden_channels = 8,
    hidden_kern = 5,
    input_kern = 3,
    layers = 6,
    gamma_hidden = 0.02,
    gamma_input = 0.2,
    depth_separable = True,
    init_sigma_range= 0.1,
    use_avg_reg=False, 
    data_info=None,
)



data = pickleread('/project/data_all_mei_and_ori.pickle') #<- filter these: 
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
    pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
    ori.append(data[i]['preferred_ori']/180)
pos1 = torch.Tensor(pos1)[ids0]
pos2 = torch.Tensor(pos2)[ids0]
ori = torch.Tensor(ori)[ids0]


model = BRCNN_with_common_scaling(dataloaders, seed=0, **model_config)
model.train()
model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori

val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)

# %%


#%%
sd = model.state_dict()
plt.plot(model.readout['all_sessions'].mu[0,0, :, 0, 0].cpu().detach(), pos1, '*')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
plt.plot(model.readout['all_sessions'].mu[0,0, :, 0, 1].cpu().detach(), pos2, '*')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
plt.plot(model.readout['all_sessions'].mu[0,0, :, 0, 2].cpu().detach(), ori, '*')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
#%%

trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 200,
    'lr_init':  0.001,
    'lr_decay_steps': 4,
    'patience': 5,
    'track_training':True,
    'verbose': True,
    'adamw': True, 
    'fine_tune': 'readout',
    })
trainer = get_trainer(trainer_fn, trainer_config)


dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
    {'dataset': 'CSRF19_V1',
    'neuronal_data_files': subject_1_filenames,
    'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
    'crop': 70,
    'subsample': 1,
    'seed': 1000,
    'scale': 0.5,
    'time_bins_sum': 12,
    'batch_size': 8, 
    'normalize_resps':True})

dataloaders1 = get_data(dataset_fn, dataset_config)

model = BRCNN_with_common_scaling(dataloaders1, seed=0, **model_config)
model.train()
model.core.load_state_dict(sd, strict=False)

data = pickleread('/project/data_all_mei_and_ori.pickle') #<- filter these: 
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
    pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
    ori.append(data[i]['preferred_ori']/180)
pos1 = torch.Tensor(pos1)[ids1]
pos2 = torch.Tensor(pos2)[ids1]
ori = torch.Tensor(ori)[ids1]
model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori

val_correlation, output, model_state_dict = trainer(model, dataloaders1,  seed=seed)

#%%
model = BRCNN_with_common_scaling(dataloaders, seed=0, **model_config)
model.load_state_dict(sd)


print(get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False))
#%%

#%%
data = pickleread('/project/data_all_mei_and_ori.pickle') #<- filter these: 
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
    pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
    ori.append(data[i]['preferred_ori']/180)
pos1 = torch.Tensor(pos1)[ids0]
pos2 = torch.Tensor(pos2)[ids0]
ori = torch.Tensor(ori)[ids0]

model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori
model.eval()
print(get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False))

#%%

#%%



# %%
model.readout
# %%

# %%
