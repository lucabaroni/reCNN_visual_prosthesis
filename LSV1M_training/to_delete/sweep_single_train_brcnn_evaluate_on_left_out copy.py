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
import torch
import os
import numpy as np
from models import BRCNN_no_scaling
from nnfabrik.builder import get_trainer
from nnvision.utility.measures import get_correlations
from LSV1M_training.dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders
from LSV1M_training.load_best_brcnn import (
    set_random_seeds,
    get_neuron_indices,
    get_population_indices,
    get_positions_and_orientations
)

# Ensure CUDA device 1 is available and set it as default
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() < 2:
    raise RuntimeError("CUDA device 1 is not available. Only {} device(s) found.".format(torch.cuda.device_count()))

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)  # Set default CUDA device
else: 
    device = torch.device('cpu')
    print("CUDA is not available")
def setup_wandb(config):
    """Initialize wandb with provided config"""
    wandb.init(project="LSV1M_brcnn_training", config=config)

def get_base_config():
    """Get base configuration for training"""
    # Dataset config
    dataset_config = {
        'population': 'both',
        'batch_size': 16,
        'n_exc': 20000,
        'n_inh': 5000,
        'n_images_val': 5000,
    }
    
    # Training config
    training_config = {
        'lr_init': 0.001,
        'lr_decay_steps': 3,
        'patience': 4,
        'adamW': False,
        'seed': 1,
    }
    
    # Model architecture config
    architecture_config = {
        'upsampling': 1,
        'rot_eq_batch_norm': True,
        'input_kern': 5,
        'hidden_kern': 5,
        'hidden_channels': 8,
        'depth_separable': True,
        'gamma_hidden': 0,
        'gamma_input': 0,
        'init_sigma_range': 0.1,
        'layers': 5,
    }
    
    return {**dataset_config, **training_config, **architecture_config}

def get_trainer_config(config):
    """Configure trainer settings"""
    trainer_config = {
        # Core training settings
        'device': device,  # Pass torch.device object instead of string
        'max_iter': 1000,
        'lr_init': config['lr_init'],
        'lr_decay_steps': config['lr_decay_steps'],
        'patience': config['patience'],
        'adamw': config['adamW'],
        
        # Loss and optimization settings
        'stop_function': 'get_correlations',
        'loss_function': 'MSE',
        'maximize': True,
        'avg_loss': False,
        
        # Other settings
        'track_training': False,
        'verbose': True,
        'fine_tune': 'core'
    }
    wandb.config.update({'trainer': trainer_config})
    return trainer_config

def get_brcnn_model_config(config):
    """Get model configuration for BRCNN"""
    model_config = {
        # Core architecture settings
        'num_rotations': 32,
        'stride': 1,
        'layers': config['layers'],
        'hidden_channels': config['hidden_channels'],
        
        # Kernel settings
        'input_kern': config.get('input_kern', 3),
        'hidden_kern': config['hidden_kern'],
        
        # Regularization settings
        'input_regularizer': 'LaplaceL2norm',
        'gamma_hidden': config['gamma_hidden'],
        'gamma_input': config['gamma_input'],
        'use_avg_reg': False,
        
        # Feature settings
        'upsampling': config.get('upsampling', 1),
        'rot_eq_batch_norm': config.get('rot_eq_batch_norm', True),
        'depth_separable': config['depth_separable'],
        'init_sigma_range': config['init_sigma_range'],
        
        # Training behavior
        'do_not_sample': True,
        'freeze_positions_and_orientations': True
    }
    wandb.config.update({'model': model_config})
    return model_config

def train_base_model(config, trainer, dataloaders, model_config):
    """Train and evaluate base model"""
    model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    model.to(device)

    # Get positions and orientations
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices('0')
    neuron_idxs = get_population_indices(config['population'], exc_neuron_idxs, inh_neuron_idxs)
    pos1, pos2, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', neuron_idxs)

    # Set positions for selected neurons
    readout = model.readout['all_sessions'].mu.data
    readout[0, 0, :, 0, 0] = pos1 / 5.5
    readout[0, 0, :, 0, 1] = -pos2 / 5.5
    readout[0, 0, :, 0, 2] = ori / np.pi

    # Train model
    val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=config['seed'])
    test_corr = get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=True)

    wandb.log({
        'val/corr': output['validation_corr'],
        'test/avg_corr': test_corr.mean(),
        'test/corr_per_neuron': test_corr
    })

    return model, model_state_dict, test_corr

def save_core_state_dict(model):
    """Save core state dict as wandb artifact"""
    core_sd = {k: v for k, v in model.state_dict().items() if k.startswith('core')}
    artifact = wandb.Artifact('core_sd', type='model')
    core_sd_path = 'core_sd.pth'
    torch.save(core_sd, core_sd_path)
    artifact.add_file(core_sd_path)
    wandb.log_artifact(artifact)
    return core_sd

def evaluate_left_out(config, model_config, core_sd):
    """Evaluate model on left out neurons"""
    # Get indices for left out neurons
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices('1')
    neuron_idxs = get_population_indices(config['population'], exc_neuron_idxs, inh_neuron_idxs)

    # Get dataloaders for left out neurons
    dataloaders_left = get_LSV1M_dataloaders(
        population=config['population'],
        batch_size=config['batch_size'],
        n_images_val=config['n_images_val'],
        exc_neuron_idxs=exc_neuron_idxs,
        inh_neuron_idxs=inh_neuron_idxs,
        center_input=False, 
    )

    # Initialize and load model
    model = BRCNN_no_scaling(dataloaders_left, seed=0, **model_config)
    model.load_state_dict(core_sd, strict=False)
    model.to(device)

    # Get positions and orientations
    pos1, pos2, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', neuron_idxs)

    # Set positions for left out neurons
    readout = model.readout['all_sessions'].mu.data
    readout[0, 0, :, 0, 0] = pos1 / 5.5
    readout[0, 0, :, 0, 1] = -pos2 / 5.5
    readout[0, 0, :, 0, 2] = ori / np.pi

    # Evaluate
    test_corr = get_correlations(model, dataloaders_left['test'], as_dict=False, per_neuron=True)
    
    wandb.log({
        'test/avg_corr_left_out_neurons': test_corr.mean(),
        'test/avg_corr_left_out_neurons_per_neuron': test_corr
    })

    return model, test_corr

def main():

    # Print CUDA device information
    print(f"Using CUDA device {device}")
    print(f"CUDA device name: {torch.cuda.get_device_name(device)}")  # Pass device index directly instead of device object
    
    # Initialize wandb with base config
    base_config = get_base_config()
    setup_wandb(base_config)
    config = {**base_config, **wandb.config}
    
    # Get all configurations
    trainer_config = get_trainer_config(config)
    model_config = get_brcnn_model_config(config)

    # Setup training
    set_random_seeds(config['seed'])
    trainer = get_trainer('nnvision.training.trainers.nnvision_trainer', trainer_config)

    # Get dataloaders for base model
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices('0')

    dataloaders = get_LSV1M_dataloaders(
        population=config['population'],
        batch_size=config['batch_size'],
        n_images_val=config['n_images_val'],
        exc_neuron_idxs=exc_neuron_idxs,
        inh_neuron_idxs=inh_neuron_idxs, 
        center_input=False,
    )
  

    # Train base model
    model, model_state_dict, test_corr = train_base_model(config, trainer, dataloaders, model_config)

    
    # Save core state dict
    core_sd = save_core_state_dict(model)


    # Evaluate on left out neurons
    model_left, test_corr_left = evaluate_left_out(config, model_config, core_sd)

    wandb.finish()

if __name__ == "__main__":
    main()

