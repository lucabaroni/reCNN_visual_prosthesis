#%%
import yaml
import wandb
from nnfabrik.builder import get_trainer, get_data
import torch
import matplotlib.pyplot as plt
import numpy as np 
import random 
import datajoint as dj 
import os 
from pickle_utils import pickleread
from nnvision.utility.measures import get_avg_correlations, get_correlations, get_MSE
from models import BRCNN_no_scaling
import h5py

dj.config["enable_python_native_blobs"] = True
dj.config["database.host"] = os.environ["DJ_HOST"]
dj.config["database.user"] = os.environ["DJ_USER"]
dj.config["database.password"] = os.environ["DJ_PASSWORD"]

# Set up your default hyperparameters
with open("/project/monkey_training/sweep_config_test.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

run = wandb.init(config=config)

lr_init = wandb.config.lr_init
batch_size = wandb.config.batch_size
upsampling= wandb.config.upsampling
rot_eq_batch_norm = wandb.config.rot_eq_batch_norm 
hidden_channels = wandb.config.hidden_channels
hidden_kern = wandb.config.hidden_kern 
input_kern = wandb.config.input_kern
layers = wandb.config.layers 
gamma_hidden = float(wandb.config.gamma_hidden)
gamma_input = float(wandb.config.gamma_input)
depth_separable = wandb.config.depth_separable
init_sigma_range = 0.1
fine_tune = 'core'

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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


trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer',
    {'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 200,
    'lr_init':  lr_init,
    'lr_decay_steps': 3, 
    'patience': 4,
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
    'normalize_resps':True})
dataloaders = get_data(dataset_fn, dataset_config)

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

#%%
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
    do_not_sample=True, 
    freeze_positions_and_orientations=True,
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda().train()
model.readout['all_sessions'].mu.data[0,0,:,0,0] = id0_pos1
model.readout['all_sessions'].mu.data[0,0,:,0,1]  = id0_pos2
model.readout['all_sessions'].mu.data[0,0,:,0,2]  = id0_ori

pos_init = model.readout['all_sessions'].mu.data[0,0,:,0,0]

# train model 
val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)
#%%
results = {
    'val/corr' : output['validation_corr'], 
    'val/MSE' : get_MSE(model, dataloaders['validation'], as_dict=False, per_neuron=False), 
    'test/MSE': get_MSE(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/corr': get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
    'test/avg_corr': get_avg_correlations(model, dataloaders['test'], as_dict=False, per_neuron=False),
}
results
#%%
pos_after = model.readout['all_sessions'].mu.data[0,0,:,0,0]

assert (pos_init == pos_after).all()

#%%
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
#%%
state_dict = model.state_dict()
core_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('core'):
        core_state_dict[k] = v

#%%
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

# Log the final results to wandb
wandb.log(results)

# Log results as a summary
run.summary['val/corr'] = results['val/corr']
run.summary['val/MSE'] = results['val/MSE']
run.summary['test/MSE'] = results['test/MSE']
run.summary['test/corr'] = results['test/corr']
run.summary['test/avg_corr'] = results['test/avg_corr']
run.summary['test/avg_corr_second_subject'] = results['test/avg_corr_second_subject']
run.summary['test/corr__second_subject'] = results['test/corr__second_subject']


# Finish the wandb run
run.finish()
