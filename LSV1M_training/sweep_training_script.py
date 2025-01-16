#%%
import wandb
import torch
import numpy as np
import random
from models import BRCNN_no_scaling
from pickle_utils import pickleread
from nnfabrik.builder import get_trainer
from nnvision.utility.measures import get_correlations
from LSV1M_training.dataset import get_LSV1M_dataloaders

# Initialize Wandb
wandb.init()

# Read configuration from Wandb
config = wandb.config

# Set random seeds
seed = config.seed if 'seed' in config else 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set hyperparameters
lr_init = config.lr_init
batch_size = config.batch_size
upsampling = config.upsampling if 'upsampling' in config else 1
rot_eq_batch_norm = config.rot_eq_batch_norm if 'rot_eq_batch_norm' in config else True
hidden_channels = config.hidden_channels
hidden_kern = config.hidden_kern
input_kern = config.input_kern if 'input_kern' in config else 3
layers = config.layers
gamma_hidden = config.gamma_hidden
gamma_input = config.gamma_input
depth_separable = config.depth_separable
init_sigma_range = config.init_sigma_range
population = config.population if 'population' in config else 'both'
adamW = config.adamW if 'adamW' in config else False
n_exc = config.n_exc if 'n_exc' in config else 20000
n_inh = config.n_inh if 'n_inh' in config else 5000
lr_decay_steps = config.lr_decay_steps if 'lr_decay_steps' in config else 3
patience = config.patience if 'patience' in config else 4
fine_tune = 'core'
do_not_sample = True
freeze_positions_and_orientations = True

exc_neuron_idxs = np.arange(0, n_exc)
inh_neuron_idxs = np.arange(0, n_inh)

# Get dataloaders
dataloaders = get_LSV1M_dataloaders(
    population=population,
    batch_size=batch_size,
    n_images_val=5000,
    exc_neuron_idxs=exc_neuron_idxs,
    inh_neuron_idxs=inh_neuron_idxs,
    # center_input=False # this was not used in the training of the first model.. needs to be uncommented to fix. 
)

# Trainer configuration
trainer_fn, trainer_config = ('nnvision.training.trainers.nnvision_trainer', {
    'stop_function': 'get_correlations',
    'loss_function': 'MSE',
    'maximize': True,
    'avg_loss': False,
    'device': 'cuda',
    'max_iter': 1000,
    'lr_init': lr_init,
    'lr_decay_steps': lr_decay_steps,
    'patience': patience,
    'track_training': False,
    'verbose': True,
    'adamw': adamW,
    'fine_tune': fine_tune
    })

trainer = get_trainer(trainer_fn, trainer_config)

# Model configuration
model_config = dict(
    num_rotations=32,
    stride=1,
    input_regularizer='LaplaceL2norm',
    upsampling=upsampling,
    rot_eq_batch_norm=rot_eq_batch_norm,
    hidden_channels=hidden_channels,
    hidden_kern=hidden_kern,
    input_kern=input_kern,
    layers=layers,
    gamma_hidden=gamma_hidden,
    gamma_input=gamma_input,
    depth_separable=depth_separable,
    init_sigma_range=init_sigma_range,
    use_avg_reg=False,
    do_not_sample=do_not_sample,
    freeze_positions_and_orientations=freeze_positions_and_orientations,
)

model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
model.cuda()

# Load pos and ori
neuron_idxs = np.concatenate([exc_neuron_idxs, inh_neuron_idxs + 37500])
pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']])) / (5.5)
pos2 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']])) / (5.5)
ori = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']]) / np.pi)
model.readout['all_sessions'].mu.data[0, 0, :, 0, 0] = pos1[neuron_idxs]
model.readout['all_sessions'].mu.data[0, 0, :, 0, 1] = -pos2[neuron_idxs]
model.readout['all_sessions'].mu.data[0, 0, :, 0, 2] = ori[neuron_idxs]

# Train and evaluate the model
val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)
test_corr = get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=True)

# Log results in Wandb as summary
results = {
    'val/corr': output['validation_corr'],
    'test/avg_corr': test_corr.mean(),
    'test/corr_per_neuron': test_corr,
}

# Save core_sd as artifact
core_sd = {k: v for k, v in model.state_dict().items() if k.startswith('core')}
artifact = wandb.Artifact('core_sd', type='model')
core_sd_path = 'core_sd.pth'
torch.save(core_sd, core_sd_path)
artifact.add_file(core_sd_path)
wandb.log_artifact(artifact)

left_out_exc_neuron_idxs = np.arange(n_exc, 37500)
left_out_inh_neuron_idxs = np.arange(n_inh, 37500/4)

# Get dataloaders
left_out_dataloaders = get_LSV1M_dataloaders(
    population=population,
    batch_size=batch_size,
    n_images_val=5000,
    exc_neuron_idxs=left_out_exc_neuron_idxs,
    inh_neuron_idxs=left_out_inh_neuron_idxs
)

model2 = BRCNN_no_scaling(left_out_dataloaders, seed=0, **model_config)
model.load_state_dict(core_sd, strict=False)
model2.cuda()

# Load pos and ori
left_out_neuron_idxs = np.concatenate([left_out_exc_neuron_idxs, left_out_inh_neuron_idxs + 37500])
model2.readout['all_sessions'].mu.data[0, 0, :, 0, 0] = pos1[left_out_neuron_idxs]
model2.readout['all_sessions'].mu.data[0, 0, :, 0, 1] = -pos2[left_out_neuron_idxs]
model2.readout['all_sessions'].mu.data[0, 0, :, 0, 2] = ori[left_out_neuron_idxs]

left_out_test_corr = get_correlations(model2, left_out_dataloaders['test'], as_dict=False, per_neuron=True)
# Log results in Wandb as summary
results = {
    'test/avg_corr_left_out_neurons': left_out_test_corr.mean(),
    'test/avg_corr_left_out_neurons_per_neuron': left_out_test_corr
}

wandb.log(results)
for key, value in results.items():
    wandb.run.summary[key] = value

# Finalize Wandb run
wandb.finish()

# %%
