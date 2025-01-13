#%%
from nnfabrik.builder import get_model, get_trainer #get_data
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
'device': 'cpu',
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
    'batch_size': 64, 
    'normalize_resps': False})

dataloaders = monkey_static_loader_combined_modified(pickleread('/project/monkey_training/data_scale0.5.pkl'), **dataset_config)


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
model_fn, model_config = ('models.reCNN_bottleneck_MultiCyclicGauss3d_no_scaling',config)

model = get_model(model_fn, model_config, dataloaders=dataloaders, seed = 42)
model.cuda().train()
model.readout['all_sessions'].mu[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu[0,0,:,0,2]  = ori


#%%
trainer(model, dataloaders, seed=seed)
torch.save(model.state_dict(), f"encoder_new_test.pt")

#%%

# model.load_state_dict(torch.load( f"/project/encoder_3nd_test_longer_no_bias.pt"))

from nnvision.utility.measures import get_correlations, get_avg_correlations
from nnfabrik.builder import get_model, get_trainer, get_data
# from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext, v1_convnext_ensemble, color_v1_convnext_ensemble

avg_corr = get_avg_correlations(model, dataloaders, 'cuda')
plt.hist(avg_corr)
plt.title(avg_corr.mean())
plt.show()
# picklesave('/project/experiment_data/convnext/avg_corr.pkl', avg_corr)

from experiments.MEI import plot_img, train_mei, StandardizeClip
vmin = -1.7919
vmax = 2.1919
mean = (vmin+vmax)/2
model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0, 0, 0])
for i in range(1):
    model.eval()
    mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
    plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0.5, 0, 0])
# for i in range(1):
#     model.eval()
#     mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
#     plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0.5, 0, 0.5])
for i in range(1):
    model.eval()
    mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
    plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)
# %%
# %%
model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0.5, 0, 0.5])
for i in range(1):
    model.eval()
    mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
    plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# %%
for x in np.linspace(0, 1, 4):
    model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([ 0,0, x])
    for i in range(1):
        model.eval()
        mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.3, steps=100, size=[46,46])
        plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# %%
mean = (vmin+vmax)/2
img = next(iter(dataloaders['train']['all_sessions'])).inputs[0].reshape(1,1,46,46)-mean
plot_img(img, pixel_min =-1.7919-mean, pixel_max=2.1919-mean)
std = img.std()

class gb(torch.nn.Module):
    def __init__(self, img):
        super().__init__()
        self.img = torch.nn.Parameter(torch.Tensor(img).reshape(1, 46,46))

    def forward(self,x):
        return torch.einsum('ijkl, xkl-> ix', x,self.img)

gb_model = gb(img)
# print(gb_model(img.tile(2,1,1,1)).shape)

gb_model.cuda()
gb_model.eval()
mei = train_mei(gb_model, 'cuda', neuron=i, std=std,  vmin =vmin-mean, vmax=vmax-mean,lr=0.1, steps=2000, size=[46,46])
plot_img(mei, pixel_min =-1.7919-mean, pixel_max=2.1919-mean)


# %%
