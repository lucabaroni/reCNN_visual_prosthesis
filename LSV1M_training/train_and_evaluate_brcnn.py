#%%
"""
This script trains and evaluates a Bottlenecked Rotation equivariant Convolutional Neural Network (BRCNN) model on the LSV1M dataset.

The main workflow:
1. Trains a BRCNN model on a subset of neurons (subset '0') 
2. Saves the trained core network weights
3. Evaluates the model's generalization by testing on left-out neurons (subset '1')

The model architecture uses has frozen positions and orientations during training.
"""
import wandb
import numpy as np
from models import BRCNN_no_scaling
from nnfabrik.builder import get_trainer
from LSV1M_training.LSV1M_utils import (
    set_random_seeds,
    get_positions_and_orientations,
    train_model,
    evaluate_generalization,
    setup_cuda,
    setup_wandb,
    save_model_state,
    get_train_and_left_out_dataloaders,
    get_neurons_idxs_and_pos_ori,
    check_duplicate_keys,
    set_brcnn_positions
)
from LSV1M_training.configs import (
    get_dataset_config,
    get_trainer_config,
    get_brcnn_model_config,
)

def get_base_config():
    """Get base configuration for training"""
    config = {
        'seed': 0,
        'empty': False,
        'wandb_log': True,
        # add here the parameters to edit
        'adamw': True,
        }
    return config 

# Setup
device = setup_cuda(device_id=0) 
base_config = get_base_config()
dataset_config = get_dataset_config(**base_config)
trainer_config = get_trainer_config(**base_config, device=device)
set_random_seeds(base_config['seed'])
dataloaders, dataloaders_left = get_train_and_left_out_dataloaders(
    dataset_config['population'],
    dataset_config['batch_size'],
    dataset_config['n_images_val'], 
    center_input=False,
    empty=base_config['empty'])

# Get data for training and left-out neurons
neuron_idxs, pos1, pos2, ori = get_neurons_idxs_and_pos_ori('0', dataset_config['population'])
neuron_idxs_left, pos1_left, pos2_left, ori_left = get_neurons_idxs_and_pos_ori('1', dataset_config['population'])

# get model
model_config = get_brcnn_model_config(**base_config)
model = BRCNN_no_scaling(dataloaders, seed=base_config['seed'], **model_config)
set_brcnn_positions(model, neuron_idxs, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl')

check_duplicate_keys([
    ("model_config", model_config),
    ("dataset_config", dataset_config), 
    ("trainer_config", trainer_config)
])
if base_config['wandb_log']:
    # Setup wandb and dataloaders
    setup_wandb("LSV1M_brcnn_training", {**model_config, **trainer_config, **dataset_config, **base_config}, model_type='brcnn')
# train model
trainer = get_trainer('nnvision.training.trainers.nnvision_trainer', trainer_config)
model, model_state_dict, test_corr = train_model(model, trainer, dataloaders, device, log=base_config['wandb_log'], seed=base_config['seed'])
# model_state_dict = model.state_dict()
# save model
save_model_state(
    model_state_dict=model_state_dict,
    model_config=model_config,
    trainer_config=trainer_config,
    dataset_config=dataset_config,
    base_config=base_config,
    checkpoint_dir='saved_models', 
    name_prefix='brcnn_', 
    log=base_config['wandb_log']
    )

# Evaluate on left-out neurons
model_left = BRCNN_no_scaling(dataloaders_left, seed=0, **model_config)
core_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith('core')}
set_brcnn_positions(model_left, neuron_idxs_left, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl')
model_left.load_state_dict(core_state_dict, strict=False)
model.to(device)
test_corr_left = evaluate_generalization(model_left, dataloaders_left, device, log=base_config['wandb_log'])
# save model left
save_model_state(
    model_state_dict=model_left.state_dict(),
    model_config= model_config,
    trainer_config=trainer_config,
    dataset_config=dataset_config,
    base_config=base_config,
    checkpoint_dir='saved_models', 
    name_prefix='brcnn_left_out_',
    log=base_config['wandb_log']
    )

if base_config['wandb_log']:
    wandb.finish()

# %%
