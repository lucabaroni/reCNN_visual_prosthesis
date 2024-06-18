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

%load_ext autoreload

%autoreload 2

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
{'loss_function': 'MSE',
'stop_function': 'get_MSE',
'maximize': False,
'avg_loss': False,
'device': 'cuda',
'max_iter': 200,
'lr_init': 0.005,
# lr_init': 0.00005,
'lr_decay_steps': 3,
'patience': 4,
'verbose': True,
'track_training':True,
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
    'batch_size': 128})

dataloaders = get_data(dataset_fn, dataset_config)
next(iter(dataloaders['train']['all_sessions'])).inputs.shape

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
model_fn, model_config = ('models.reCNN_bottleneck_MultiCyclicGauss3d_with_final_scaling',config)

model_config = dict(
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    core_hidden_channels = 16, 
    core_hidden_kern = 5, 
    core_input_kern = 7,
    core_layers = 4, 
    core_gamma_hidden = 0.02,
    core_gamma_input =  0.2,
    # depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,
)

model = get_model(model_fn, model_config, dataloaders=dataloaders, seed = 42)
model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori

# Custom function to enforce sign constraints
def enforce_sign_constraint(param):
    with torch.no_grad():
        param[param < 0.02] = 0.02

# Register the hook directly to the final_scale parameter
if hasattr(model, 'final_scale'):
    model.final_scale.register_hook(lambda grad: enforce_sign_constraint(model.final_scale))

trainer(model, dataloaders, seed=seed)
#%%
import torch
from datetime import datetime

# Assume 'model' is your PyTorch model
# Get the current date and time, and format it as a string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the model state dict with the formatted date-time stamp
torch.save(model.state_dict(), f"encoder_new_data_{timestamp}.pt")

#%%
# trainer(model, dataloaders, seed=seed)
# torch.save(model.state_dict(), f"encoder_4th_new_data.pt")
model.load_state_dict(torch.load('/project/monkey_training/encoder_new_data_20240524_234116.pt'))

from nnvision.utility.measures import get_correlations, get_avg_correlations
from nnfabrik.builder import get_model, get_trainer, get_data
# from nnvision.models.trained_models.v1_task_fine_tuned import v1_convnext, v1_convnext_ensemble, color_v1_convnext_ensemble

#%%
from nnvision.utility.measures import get_correlations, get_avg_correlations
# with torch.no_grad():
#     model.readout['all_sessions'].bias.data = torch.zeros_like(model.readout['all_sessions'].bias.data) + model.readout['all_sessions'].bias.data.mean()

avg_corr = get_avg_correlations(model, dataloaders, 'cuda')
plt.hist(avg_corr)
plt.title(avg_corr.mean())
plt.show()
# picklesave('/project/experiment_data/convnext/avg_corr.pkl', avg_corr)
avg_corr = get_correlations(model, dataloaders, 'cuda')
plt.hist(avg_corr)
plt.title(avg_corr.mean())
plt.show()

#%%

pixel_min = -1.7986
pixel_max = 2.1919
mean = (pixel_min + pixel_max)/2
import featurevis
from featurevis.utils import Compose
import featurevis.ops as ops
import torch.nn as nn
from experiments.MEI import plot_img
class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()

model0 = SingleCellModel(model, 0)
def create_mei(model, std=0.05, seed=42, img_res = [93,93], pixel_min = -1.7876, pixel_max = 2.1919, gaussianblur=1., device=None, step_size=10, num_steps=1000):
    if device==None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    initial_image = torch.randn(1, 1, *img_res, dtype=torch.float32).to(device)*std
    model.eval()
    model.to(device)
    initial_image = initial_image.to(device)
    
    # TODO decide if we need to add it 
    post_update =Compose([ops.ChangeStats(std=std, mean=mean), ops.ClipRange(pixel_min, pixel_max)])
    opt_x, fevals, reg_values = featurevis.gradient_ascent(
        model,
        initial_image, 
        step_size=step_size,
        num_iterations=num_steps, 
        post_update=post_update,
        gradient_f = ops.GaussianBlur(gaussianblur),
        print_iters=1001,
    )
    mei = opt_x.detach().cpu().numpy().squeeze()
    mei_act = fevals[-1]
    return mei,  mei_act

mei, act = create_mei(model0, std=0.15, img_res=[46,46])
plot_img(mei, pixel_min, pixel_max)

#%%

plt.imshow(mei, vmin=pixel_min,vmax=pixel_max, cmap='Greys_r')
plt.axis('off')
plt.savefig('mei_brcnn.png', dpi=300)
plt.show()
# #%%

#%%
# #%%
#%%

# from experiments.MEI import plot_img, train_mei, StandardizeClip
# vmin = -1.7919
# vmax = 2.1919
# mean = (vmin+vmax)/2
# model.readout['all_sessions'].mu.data[0,0,0,0] = torch.Tensor([0, 0, 0])
# for i in range(1):
#     model.eval()
#     mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
#     plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# # model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0.5, 0, 0])
# # for i in range(1):
# #     model.eval()
# #     mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
# #     plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# model.readout['all_sessions'].mu.data[0,0,0,0] = torch.Tensor([0.5, 0, 0.5])
# for i in range(2):
#     model.eval()
#     mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
#     plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)
# # %%
# # %%
# model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([0.5, 0, 0.5])
# for i in range(1):
#     model.eval()
#     mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.1, steps=1000, size=[46,46])
#     plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# # %%
# for x in np.linspace(0, 1, 4):
#     model.readout['all_sessions'].mu[0,0,0,0] = torch.Tensor([ 0,0, x])
#     for i in range(1):
#         model.eval()
#         mei = train_mei(model, 'cuda', neuron=i, std=0.1,  vmin =vmin, vmax=vmax,lr=0.3, steps=100, size=[46,46])
#         plot_img(mei, pixel_min =-1.7919, pixel_max=2.1919)

# # %%
# mean = (vmin+vmax)/2
# img = next(iter(dataloaders['train']['all_sessions'])).inputs[0].reshape(1,1,46,46)-mean
# plot_img(img, pixel_min =-1.7919-mean, pixel_max=2.1919-mean)
# std = img.std()

# class gb(torch.nn.Module):
#     def __init__(self, img):
#         super().__init__()
#         self.img = torch.nn.Parameter(torch.Tensor(img).reshape(1, 46,46))

#     def forward(self,x):
#         return torch.einsum('ijkl, xkl-> ix', x,self.img)

# gb_model = gb(img)
# # print(gb_model(img.tile(2,1,1,1)).shape)

# gb_model.cuda()
# gb_model.eval()
# mei = train_mei(gb_model, 'cuda', neuron=i, std=std,  vmin =vmin-mean, vmax=vmax-mean,lr=0.1, steps=2000, size=[46,46])
# plot_img(mei, pixel_min =-1.7919-mean, pixel_max=2.1919-mean)


# # %%

# # %% check how positions are assigned






# %%
