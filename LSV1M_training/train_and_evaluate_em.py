#%%
"""
This script trains and evaluates an Energy Model on the LSV1M dataset.

The main workflow:
1. Trains an Energy Model on a subset of neurons (subset '0') 
2. Saves the trained core network weights
3. Evaluates the model's generalization by testing on left-out neurons (subset '1')

The model architecture uses has frozen positions and orientations during training.
"""
import wandb
from models import EnergyModel
from nnfabrik.builder import get_trainer
from LSV1M_training.LSV1M_utils import (
    set_random_seeds,
    train_model,
    evaluate_generalization,
    setup_cuda,
    setup_wandb,
    save_model_state,
    get_train_and_left_out_dataloaders,
    get_neurons_idxs_and_pos_ori,
    check_duplicate_keys
)
from LSV1M_training.configs import (
    get_dataset_config,
    get_trainer_config,
    get_energy_model_config,
)

def get_base_config():
    """Get base configuration for training"""
    config = {
        'seed': 0,
        'empty': False,
        'wandb_log': True,
        # add here the parameters to edit
        }
    return config 

# Setup
device = setup_cuda() 
base_config = get_base_config()
dataset_config = get_dataset_config(**base_config)
trainer_config = get_trainer_config(**base_config)
set_random_seeds(base_config['seed'])
dataloaders, dataloaders_left = get_train_and_left_out_dataloaders(
    dataset_config['population'],
    dataset_config['batch_size'],
    dataset_config['n_images_val'], 
    center_input=True,
    empty=base_config['empty'])

# Get data for training and left-out neurons
neuron_idxs, pos1, pos2, ori = get_neurons_idxs_and_pos_ori('0', dataset_config['population'])
neuron_idxs_left, pos1_left, pos2_left, ori_left = get_neurons_idxs_and_pos_ori('1', dataset_config['population'])

# get model
model_config = get_energy_model_config(pos1, pos2, ori) 
model = EnergyModel(dataloaders, seed=0, **model_config)

check_duplicate_keys([
    ("model_config", model_config),
    ("dataset_config", dataset_config), 
    ("trainer_config", trainer_config)
])

if base_config['wandb_log']:
    # Setup wandb and dataloaders
    setup_wandb("LSV1M_EnergyModel_training", {**model_config, **trainer_config, **dataset_config, **base_config}, model_type='em')

# train model
trainer = get_trainer('nnvision.training.trainers.nnvision_trainer', trainer_config)
model, model_state_dict, test_corr = train_model(model, trainer, dataloaders, device, log=base_config['wandb_log'])
# model_state_dict = model.state_dict()
# save model
save_model_state(
    model_state_dict=model_state_dict,
    model_config=model_config,
    trainer_config=trainer_config,
    dataset_config=dataset_config,
    base_config=base_config,
    checkpoint_dir='saved_models', 
    name_prefix='em_', 
    log=base_config['wandb_log']
    )

# Evaluate on left-out neurons
model_config_left = get_energy_model_config(pos1_left, pos2_left, ori_left) 
model_left = EnergyModel(dataloaders_left, seed=0, **model_config_left)
shared_params_state_dict = {k: v for k, v in model_state_dict.items() if k not in ['positions_x', 'positions_y', 'orientations', 'meshgrid_x', 'meshgrid_y']}
model_left.load_state_dict(shared_params_state_dict, strict=False)
test_corr_left = evaluate_generalization(model_left, dataloaders_left, device, log=base_config['wandb_log'])
# save model left
save_model_state(
    model_state_dict=model_left.state_dict(),
    model_config=model_config_left,
    trainer_config=trainer_config,
    dataset_config=dataset_config,
    base_config=base_config,
    checkpoint_dir='saved_models', 
    name_prefix='em_left_out_', 
    log=base_config['wandb_log']
    )

if base_config['wandb_log']:
    wandb.finish()


# %%
