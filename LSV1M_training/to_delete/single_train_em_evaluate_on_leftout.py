#%%
"""
This script trains and evaluates an Energy Model on the LSV1M dataset.

The main workflow:
1. Trains an Energy Model on a subset of neurons (subset '0')
2. Saves the trained model weights 
3. Evaluates the model's generalization by testing on left-out neurons (subset '1')

The model architecture uses fixed positions and orientations during training.
"""

import torch
import wandb
import os
from models import EnergyModel
from nnfabrik.builder import get_trainer
from nnvision.utility.measures import get_correlations
from LSV1M_training.dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders
from LSV1M_training.load_best_brcnn import (
    set_random_seeds,
    get_neuron_indices,
    get_population_indices,
    get_positions_and_orientations
)

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_base_config():
    """Get base configuration for training"""
    return {
        'population': 'both',
        'batch_size': 16,
        'lr_init': 0.001,
        'lr_decay_steps': 3,
        'patience': 5,
        'adamW': True,
        'n_exc': 20000,
        'n_inh': 5000,
        'n_images_val': 5000,
        'seed': 1
    }

def get_trainer_config(config):
    """Configure trainer settings"""
    return {
        'stop_function': 'get_correlations',
        'loss_function': 'MSE',
        'maximize': True,
        'avg_loss': False,
        'device': device,
        'max_iter': 10,
        'lr_init': config['lr_init'],
        'lr_decay_steps': config['lr_decay_steps'],
        'patience': config['patience'],
        'track_training': False,
        'verbose': True,
        'adamw': config['adamW']
    }

def get_energy_model_config(pos1, pos2, ori):
    """Get model configuration for energy model"""
    return {
        'positions_x': pos1,
        'positions_y': pos2,
        'orientations': ori,
        'nonlinearity': 'square_root',
        'pnl_vmin': -10,
        'pnl_vmax': 20,
        'pnl_nbis': 100,
        'pnl_smooth_reg_weight': 0,
        'pnl_smoothness_reg_order': 2,
        'resolution': [55, 55],
        'xlim': (-5.5, 5.5),
        'ylim': (-5.5, 5.5),
        'sigma_x_init': .36,
        'sigma_y_init': .4,
        'f_init': 0.7,
        'filter_scale_init': 0.01,
        'final_scale_init': 1,
        'keep_fixed_pos_and_ori': True,
        'data_info': None,
        'common_rescaling': True
    }

def setup_wandb(project_name, model_config, trainer_config):
    """Initialize and configure wandb"""
    wandb.init(project=project_name)
    wandb.config.update({**model_config, **trainer_config})

def save_model(model_state_dict, model_config, checkpoint_dir, name_prefix=""):
    """Save model checkpoint and log to wandb"""
    checkpoint_path = os.path.join(checkpoint_dir, f'{name_prefix}model_checkpoint.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': model_state_dict,
        'config': model_config
    }, checkpoint_path)
    
    artifact = wandb.Artifact(f'{name_prefix}model_checkpoint', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)

def train_base_model(config, trainer, dataloaders, model_config):
    """Train and evaluate base model"""
    model = EnergyModel(dataloaders, seed=0, **model_config)
    model.to(device)
    
    _, output, _ = trainer(model, dataloaders, seed=config['seed'])
    model_state_dict = model.state_dict()
    test_corr = get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=True)
    
    wandb.log({
        'val/corr': output['validation_corr'],
        'test/avg_corr': test_corr.mean(),
        'test/corr_per_neuron': test_corr
    })
    
    return model, model_state_dict, test_corr

def evaluate_left_out(model_state_dict, model_config, dataloaders_left):
    """Evaluate model on left out neurons"""
    model_left = EnergyModel(dataloaders_left, seed=0, **model_config)
    model_left.to(device)
    
    # Remove position-specific parameters
    model_state_dict_left = model_state_dict.copy()
    for key in ['positions_x', 'positions_y', 'orientations', 'meshgrid_x', 'meshgrid_y']:
        model_state_dict_left.pop(key)
    
    model_left.load_state_dict(model_state_dict_left, strict=False)
    test_corr_left = get_correlations(model_left, dataloaders_left['test'], as_dict=False, per_neuron=True)
    
    print(f"Average correlation on left out neurons: {test_corr_left.mean():.3f}")
    
    wandb.log({
        'left_out/avg_corr': test_corr_left.mean(),
        'left_out/corr_per_neuron': test_corr_left
    })
    
    return model_left, test_corr_left

def main():
    # Get configurations
    config = get_base_config()
    trainer_config = get_trainer_config(config)
    trainer = get_trainer('nnvision.training.trainers.nnvision_trainer', trainer_config)

    # Set random seeds
    set_random_seeds(config['seed'])

    # Get indices for base model
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices('0')
    neuron_idxs = get_population_indices(config['population'], exc_neuron_idxs, inh_neuron_idxs)

    # Get positions and configurations
    pos2, pos1, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', neuron_idxs)
    pos1, pos2, ori = -pos1, pos2, -ori
    model_config = get_energy_model_config(pos1, pos2, ori)

    # Get dataloaders for base model
    dataloaders = get_LSV1M_dataloaders(
        population=config['population'],
        batch_size=config['batch_size'],
        n_images_val=config['n_images_val'],
        exc_neuron_idxs=exc_neuron_idxs,
        center_input=True,
        inh_neuron_idxs=inh_neuron_idxs
    )

    # Initialize wandb
    setup_wandb("LSV1M_energy_model_training", model_config, trainer_config)

    # Train base model
    model, model_state_dict, test_corr = train_base_model(config, trainer, dataloaders, model_config)
    save_model(model_state_dict, model_config, 'checkpoints')

    # Get indices for left out neurons
    exc_neuron_idxs_left, inh_neuron_idxs_left = get_neuron_indices('1')
    neuron_idxs_left = get_population_indices(config['population'], exc_neuron_idxs_left, inh_neuron_idxs_left)

    # Get positions and orientations for left out neurons
    pos2_left, pos1_left, ori_left = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', neuron_idxs_left)
    pos1_left, pos2_left, ori_left = -pos1_left, pos2_left, -ori_left

    # Get dataloaders for left out neurons
    dataloaders_left = get_LSV1M_dataloaders(
        population=config['population'],
        batch_size=config['batch_size'],
        n_images_val=config['n_images_val'],
        exc_neuron_idxs=exc_neuron_idxs_left,
        center_input=True,
        inh_neuron_idxs=inh_neuron_idxs_left
    )

    # Configure and evaluate left out model
    model_config_left = {**model_config, **{
        'positions_x': pos1_left,
        'positions_y': pos2_left,
        'orientations': ori_left
    }}

    model_left, test_corr_left = evaluate_left_out(model_state_dict, model_config_left, dataloaders_left)
    save_model(model_left.state_dict(), model_config_left, 'checkpoints', name_prefix='left_out_neurons_')

    wandb.finish()

if __name__ == "__main__":
    main()

#%%
