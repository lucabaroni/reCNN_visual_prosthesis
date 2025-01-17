#%%
import yaml
import wandb
from nnfabrik.builder import get_model, get_trainer, get_data
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 
from pickle_utils import pickleread
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE
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

run = wandb.init(project='recnn_train', config=config)

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
    'adamw': True})

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

for file in filenames: 
    n = len(pickleread(file)['training_responses']) 
    if pickleread(file)['subject_id']== 31: 
        subject_0_filenames.append(file)
        ids0 = ids0 + list(np.arange(skip, skip + n))

    else:
        subject_1_filenames.append(file)
        ids1 = ids1 + list(np.arange(skip, skip + n ))
    skip += n

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
    'normalize_resps':True, 
    'center_inputs': True})

dataloaders = get_data(dataset_fn, dataset_config)
# %%
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data1[i]['preferred_pos'][1]-(93/2))/(93/2)) # x is y in the data
    pos2.append((data1[i]['preferred_pos'][0]-(93/2))/(93/2)) # y is x in the data
    ori.append(-data1[i]['preferred_ori']) # change sign only here
id0_pos1 = torch.Tensor(pos1)[ids0]
id0_pos2 = torch.Tensor(pos2)[ids0]
id0_ori = torch.Tensor(ori)[ids0]
#%%
config=dict(
    positions_x = id0_pos1,
    positions_y = id0_pos2,
    orientations = id0_ori,
    sigma_x_init = 0.2,
    sigma_y_init= 0.2,
    f_init = 2, 
    filter_scale_init = 0.01, 
    final_scale_init = 1,
    resolution = (46,46),
    xlim = (-2.67/2, 2.67/2),
    ylim = (-2.67/2, 2.67/2), 
    pnl_vmin = -10, 
    pnl_vmax = 10, 
    pnl_nbis = 50, 
    nonlinearity='piecewise_nonlinearity', 
    common_rescaling=True,
    keep_fixed_pos_and_ori=True
)

model_fn, model_config = ('models.EnergyModel', config)
model = get_model(model_fn, model_config, dataloaders=dataloaders, seed = 42)

#%%
for i in range(0, 10):
    x=model.get_filters_even_and_odd()[0][i]
    plt.imshow(x.detach().cpu())
    plt.title(i)
    plt.show()

#%%
# train model 
val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)
#%%
results = {
    # 'val/corr' : output['validation_corr'], 
    'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/corr': get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),

}
print(results)

#%%
model_save_path = f'/project/monkey_training/energy_model.pt'
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
    if k.startswith('positions')!=True and k.startswith('orientations')!=True and  k.startswith('meshgrid')!=True:
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
    'normalize_resps':True, 
    'center_inputs': True}
    )
left_out_dataloaders = get_data(left_out_dataset_fn, left_out_dataset_config)
#%%
id1_pos1 = torch.Tensor(pos1)[ids1]
id1_pos2 = torch.Tensor(pos2)[ids1]
id1_ori = torch.Tensor(ori)[ids1]
#%%
config.update(dict(
    positions_x = id1_pos1,
    positions_y = id1_pos2,
    orientations = id1_ori))

#%%
model2 = get_model(model_fn, config, dataloaders=left_out_dataloaders, seed = 42)
model2.cuda().train()
model2.load_state_dict(core_state_dict, strict=False)

results.update({
    'test/corr_second_subject' : get_correlations(model2, left_out_dataloaders['test'], as_dict=False, per_neuron=False), 
    'test/avg_corr_second_subject' : get_avg_correlations(model2, left_out_dataloaders['test'], as_dict=False, per_neuron=False), 
})
print(results)
#%%
for i in range():
    x=model2.get_filters_even_and_odd()[0][i]
    plt.imshow(x.detach().cpu())
    plt.title(i)
    plt.show()
#%%
wandb.log(results)
wandb.run.summary.update(results)

# Finish the wandb run
run.finish()
# %%
    