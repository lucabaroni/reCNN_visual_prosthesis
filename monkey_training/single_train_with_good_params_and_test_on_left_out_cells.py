#%%
from curses import resetty
import yaml
import wandb
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
# from nnvision.datasets.monkey_loaders import monkey_static_loader_combined_modified
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE
from models import BRCNN_with_common_scaling, BRCNN_no_scaling
import h5py

dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]



seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

lr_init = 0.001
batch_size = 4
upsampling= 1
rot_eq_batch_norm = True
hidden_channels = 8
hidden_kern = 9
input_kern = 5
layers = 7
gamma_hidden = 0.0195
gamma_input = 0
depth_separable = False
init_sigma_range = 0.1
fine_tune='core'

h5 = '/project/results_convnext_model.h5'


data1 = {i: {} for i in range(458)}
with h5py.File(h5, 'r') as f:
    for i in range(458):
        dataset_name = f'full_field_params/neuron_{i}'
        if dataset_name in f:
            data1[i]['preferred_ori'] = f[dataset_name][:][0]
        dataset_name = f'preferred_pos/neuron_{i}'
        if dataset_name in f:
            data1[i]['preferred_pos'] = f[dataset_name][:]


# Set up your default hyperparameters
with open("/project/monkey_training/sweep_config_test.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(project='recnn_train_data', config=config)


trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'device': 'cuda',
    'max_iter': 200,
    'lr_init':  lr_init,
    'lr_decay_steps': 3, 
    'patience': 5,
    'track_training':False,
    'verbose': True,
    'adamw': True,
    'fine_tune': fine_tune})

trainer = get_trainer(trainer_fn, trainer_config)

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

for i, file in enumerate(filenames): 
    n = len(pickleread(file)['training_responses']) 
    if i%2==0: 
        subject_0_filenames.append(file)
        ids0 = ids0 + list(np.arange(skip, skip + n))
        print(i)
    else:
        subject_1_filenames.append(file)
        ids1 = ids1 + list(np.arange(skip, skip + n ))
    skip += n

#%%
dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
    {'dataset': 'CSRF19_V1',
    'neuronal_data_files': subject_0_filenames,
    'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
    'crop': 70,
    'subsample': 1,
    'seed': 1000,
    'scale': 0.5,
    'time_bins_sum': 12,
    'batch_size': batch_size, 
    'normalize_resps':True})
dataloaders = get_data(dataset_fn, dataset_config)


# neuron_datasets = {}
# def stack_neuron_datasets(file_name):
#     with h5py.File(file_name, 'r') as f:
#         neuron_datasets = []
    
#         def collect_neuron_datasets(name, obj):
#             if isinstance(obj, h5py.Dataset) and name.startswith('full_field_params/neuron_'):
#                 neuron_datasets[](f[name][:])
    
#         f.visititems(collect_neuron_datasets)
    
#         # Stack all collected datasets
#         if neuron_datasets:
#             stacked_data = np.vstack(neuron_datasets)
#             return stacked_data
#         else:
#             return None

# data = pickleread('/project/data_all_mei_and_ori.pickle')
# #%%
# data = pickleread('/project/data_all_mei_and_ori.pickle')
# id0_pos1 = []
# id0_pos2 = []
# id0_ori = []
# for i in range(458):
#     id0_pos1.append((data[i]['center_mask_mei'][0]-(93/2))/(93/2))
#     id0_pos2.append((data[i]['center_mask_mei'][1]-(93/2))/(93/2))
#     id0_ori.append(data[i]['preferred_ori']/180)
# id0_pos1 = torch.Tensor(id0_pos1)[ids0]
# id0_pos2 = torch.Tensor(id0_pos2)[ids0]
# id0_ori = torch.Tensor(id0_ori)[ids0]


# new 
id0_pos1 = []
id0_pos2 = []
id0_ori = []
for i in range(458):
    id0_pos1.append((data1[i]['preferred_pos'][0]-(93/2))/(93/2))
    id0_pos2.append((data1[i]['preferred_pos'][1]-(93/2))/(93/2))
    id0_ori.append(data1[i]['preferred_ori']/np.pi)
id0_pos1 = torch.Tensor(id0_pos1)[ids0]
id0_pos2 = torch.Tensor(id0_pos2)[ids0]
id0_ori = torch.Tensor(id0_ori)[ids0]

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
    data_info=None,
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = id0_pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = id0_pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = id0_ori

# train model 
val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)

results = {
    'val/corr' : output['validation_corr'], 
    'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/corr': get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),

}


model_save_path = f'/project/monkey_training/fine_tune={fine_tune}_best_model.pt'
torch.save(
    {
        'model_config': model_config,
        'model_state_dict': model.state_dict()
    },
    model_save_path
)

# Create a wandb artifact and add the model file
artifact = wandb.Artifact('best_model', type='model')
artifact.add_file(model_save_path)
# Log the artifact to wandb
run.log_artifact(artifact)

state_dict = model.state_dict()
core_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('core'):
        core_state_dict[k] = v

# test on second subject
left_out_dataset_fn, left_out_dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
    {'dataset': 'CSRF19_V1',
    'neuronal_data_files': subject_1_filenames,
    'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
    'crop': 70,
    'subsample': 1,
    'seed': 1000,
    'scale': 0.5,
    'time_bins_sum': 12,
    'batch_size': batch_size, 
    'normalize_resps':True})
left_out_dataloaders = get_data(left_out_dataset_fn, left_out_dataset_config)


id1_pos1 = []
id1_pos2 = []
id1_ori = []
for i in range(458):
    id1_pos1.append((data1[i]['preferred_pos'][0]-(93/2))/(93/2))
    id1_pos2.append((data1[i]['preferred_pos'][1]-(93/2))/(93/2))
    id1_ori.append(data1[i]['preferred_ori']/np.pi)
id1_pos1 = torch.Tensor(id1_pos1)[ids1]
id1_pos2 = torch.Tensor(id1_pos2)[ids1]
id1_ori = torch.Tensor(id1_ori)[ids1]

model2 = BRCNN_no_scaling(left_out_dataloaders, seed=0, **model_config)
model2.cuda().train()
model2.load_state_dict(core_state_dict, strict=False)

model2.readout['all_sessions'].mu.data[0,0,:,0,0] = id1_pos1
model2.readout['all_sessions'].mu.data[0,0,:,0,1]  = id1_pos2
model2.readout['all_sessions'].mu.data[0,0,:,0,2]  = id1_ori

results.update({
    'test/avg_corr_second_subject' : get_avg_correlations(model2, left_out_dataloaders['test'], as_dict=False, per_neuron=False), 
    'test/corr__second_subject' : get_correlations(model2, left_out_dataloaders['test'], as_dict=False, per_neuron=False), 
})

wandb.log(results)
wandb.run.summary.update(results)


# Finish the wandb run
run.finish()


    # #%%
    # import matplotlib.pyplot as plt


# plt.scatter(
#     [data[i]['center_mask_mei'][1] for i in range(458)], 
#     [data1[i]['preferred_pos'][1] for i in range(458)], 
#     c=[data1[i]['preferred_pos'][2] for i in range(458)], 
#     alpha=0.7, 
#     s=5
#     )
# plt.xlim(20, 70)
# plt.ylim(20, 70)
# plt.colorbar()
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title('y position')
# plt.xlabel('estimated from MEI')
# plt.ylabel('estimated from dot stimulation RF protocol')
# plt.show()


# plt.scatter(
#     [data[i]['center_mask_mei'][0] for i in range(458)], 
#     [data1[i]['preferred_pos'][0] for i in range(458)], 
#     c=[data1[i]['preferred_pos'][2] for i in range(458)], 
#     alpha=0.7, 
#     s=5
#     )
# plt.xlim(20, 70)
# plt.ylim(20, 70)
# plt.colorbar()
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title('x position')
# plt.xlabel('estimated from MEI')
# plt.ylabel('estimated from dot stimulation RF protocol')
# plt.show()

# plt.scatter(
#     x = [data[i]['center_mask_mei'][0] - data1[i]['preferred_pos'][0] for i in range(458)],
#     y = [data[i]['center_mask_mei'][1] - data1[i]['preferred_pos'][1] for i in range(458)],
#     c=[data1[i]['preferred_pos'][2] for i in range(458)], 
#     alpha=0.7, 
#     s=10
#     )
# plt.colorbar()
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title('position difference vs RF gaussian fit error (in colorbar)')
# plt.xlabel('x position difference')
# plt.ylabel('y position difference')

# plt.show()

# # %%
# plt.scatter(
#     [data[i]['preferred_ori'] + np.random.randn()*3 for i in range(458)], 
#     [data1[i]['preferred_ori']/np.pi * 180  + np.random.randn()*3 for i in range(458)], 
#     alpha=0.7, 
#     s=5
#     )

# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title('ori diff ')
# plt.xlabel('estimated previously')
# plt.ylabel('estimated with the new pipeline')
# plt.show()
# # %%
# plt.hist2d([data[i]['preferred_ori'] for i in range(458)], [data1[i]['preferred_ori']/np.pi * 180 for i in range(458)], bins=40)
# plt.xlim(0, None)
# plt.ylim(0, None)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.xlabel('estimated previously')
# plt.ylabel('estimated with the new pipeline')
# plt.colorbar()
# # %%

 # %%

# %%

import featurevis
import featurevis.ops as ops
from featurevis.utils import Compose
import torch.nn as nn


class SingleCellModel(nn.Module):
    def __init__(self, model, idx):
        super().__init__()
        self.model = model
        self.idx = idx
    
    def forward(self, x):
        return self.model(x)[:, self.idx].squeeze()


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
    mean = (pixel_min + pixel_max)/2
    
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

#%%
for i in [35, 67]:
    model0 = SingleCellModel(model, i)
    mei, act = create_mei(model0, std=0.15, img_res=[46,46])
    import matplotlib.pyplot as plt 
    plt.imshow(mei)
    plt.title(i)
    plt.show()
# %%