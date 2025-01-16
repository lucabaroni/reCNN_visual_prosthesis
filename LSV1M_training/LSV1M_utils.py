import torch
import numpy as np
import random
from pickle_utils import pickleread
import wandb
from nnvision.utility.measures import get_correlations
import os
from LSV1M_training.dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders, get_LSV1M_dataloaders_preprocessed


def set_random_seeds(seed=1):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_cuda(device_id=0):
    """
    Setup CUDA device with specified requirements.
    Returns:
        torch.device: Selected CUDA device or CPU if CUDA is not available
    """
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU")
        return torch.device('cpu')
    
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    return device


def get_neuron_indices(neurons_subset):
    """Get excitatory and inhibitory neuron indices based on subset."""
    neuron_ranges = {
        '0': (np.arange(20000), np.arange(5000)),
        '1': (np.arange(20000, 37500), np.arange(5000, 9375)), 
        'both': (np.arange(37500), np.arange(9375))
    }
    
    if neurons_subset not in neuron_ranges:
        raise ValueError("neurons_subset must be '0', '1', or 'both'")
        
    return neuron_ranges[neurons_subset]

def get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs):
    """Get combined neuron indices based on population type."""
    if population == 'both':
        return list(exc_neuron_idxs) + list(inh_neuron_idxs + 37500)
    elif population == 'exc':
        return list(exc_neuron_idxs)
    elif population == 'inh':
        return list(inh_neuron_idxs + 37500)
    else:
        raise ValueError("population must be 'both', 'exc', or 'inh'")

def get_positions_and_orientations(pos_ori_path, neuron_idxs):
    """Load and process positions and orientations for neurons."""
    pos_ori = pickleread(pos_ori_path)
    
    pos_x = np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']])
    pos_y = np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']])
    ori = np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']])
    
    pos_x = torch.Tensor(pos_x)[neuron_idxs]
    pos_y = torch.Tensor(pos_y)[neuron_idxs]
    ori = torch.Tensor(ori)[neuron_idxs]
        
    return pos_x, pos_y, ori

def get_neurons_idxs_and_pos_ori(subset, population, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl'):
    """Get neuron indices and positions for a given subset and population.
    
    Args:
        subset (str): Subset of neurons ('0' for training or '1' for left-out)
        population (str): Population type ('exc', 'inh', or 'both')
        pos_ori_path (str): Path to positions and orientations file
        
    Returns:
        tuple: (neuron_indices, pos1, pos2, ori)
    """
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices(subset)
    neuron_idxs = get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs)
    pos1, pos2, ori = get_positions_and_orientations(pos_ori_path, neuron_idxs)
    return neuron_idxs, pos1, pos2, ori


# setup function
def setup_wandb(project_name, config, model_type):
    """
    Initialize wandb with provided config.
    Args:
        project_name (str): Base project name
        config (dict): Configuration dictionary
        model_type (str): Type of model ('brcnn' or 'em')
    """
    
    full_project_name = f"LSV1M_{model_type}_training"
    wandb.init(
        project=full_project_name,
        config=config,
        name=project_name
    )

def train_model(model, trainer, dataloaders, device, log=True, seed=1):
    """Generic training function for both BRCNN and EM"""
    model.to(device)
    
    val_correlation, output, model_state_dict = trainer(model, dataloaders, seed=seed)
    test_corr = get_correlations(model, dataloaders['test'], as_dict=False, per_neuron=True)
    
    if log:
        wandb.log({
            'val/corr': output['validation_corr'],
            'test/avg_corr': test_corr.mean(),
            'test/corr_per_neuron': test_corr
        })
    
    return model, model_state_dict, test_corr

# evaluate generalization
def evaluate_generalization(model_left, dataloaders_left, device, log=True):
    """Generic evaluation function for left-out neurons"""
    model_left.to(device)
    test_corr = get_correlations(model_left, dataloaders_left['test'], as_dict=False, per_neuron=True, device=device)
    if log:
        wandb.log({
            'left_out/avg_corr': test_corr.mean(),
            'left_out/corr_per_neuron': test_corr
        })
    return test_corr

def save_model_state(model_state_dict, checkpoint_dir, name_prefix="", model_config=None, trainer_config=None, dataset_config=None, base_config=None, log=True):
    """Save model checkpoint and log to wandb"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{name_prefix}_model_checkpoint_{timestamp}.pth')
    
    # save model state dic
    d = {
        'model_config': model_config,
        'trainer_config': trainer_config,
        'base_config': base_config,
        'dataset_config': dataset_config,
        'model_state_dict': model_state_dict,
    }
    torch.save(d, checkpoint_path)
    
    if log:
        artifact = wandb.Artifact(f'{name_prefix}model_checkpoint', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

def get_train_and_left_out_dataloaders(population, batch_size, n_images_val, center_input, empty=False):
    """Get dataloaders for training and left-out neurons.
    
    This function creates two sets of dataloaders - one for training neurons and one for left-out neurons
    used for evaluating generalization. It can create either real dataloaders with actual data or empty
    ones for development purposes.

    Args:
        population (str): Which neuron population to use ('exc', 'inh', or 'both')
        batch_size (int): Batch size for the dataloaders
        n_images_val (int): Number of images to use for validation
        center_input (bool): Whether to center the input images. Should be True for Energy Models,
            False for BRCNNs.
        empty (bool): If True, creates empty dataloaders for development. Defaults to False.

    Returns:
        tuple: A pair of dataloaders:
            - dataloaders: DataLoader for training neurons (subset '0')
            - dataloaders_left: DataLoader for left-out neurons (subset '1')
    """
    
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices('0')
    exc_neuron_idxs_left, inh_neuron_idxs_left = get_neuron_indices('1')
    if empty: 
        # empty for development
        dataloaders = get_LSV1M_empty_dataloaders(
            population=population,
            batch_size=batch_size,
            n_images_val=n_images_val,
            exc_neuron_idxs=exc_neuron_idxs, 
            inh_neuron_idxs=inh_neuron_idxs)
        dataloaders_left = get_LSV1M_empty_dataloaders(
            population=population,
            batch_size=batch_size,
            n_images_val=n_images_val,
            exc_neuron_idxs=exc_neuron_idxs_left, 
            inh_neuron_idxs=inh_neuron_idxs_left)
    else: 
        dataloaders = get_LSV1M_dataloaders_preprocessed(
            population=population,
            batch_size=batch_size,
            n_images_val=n_images_val,
            exc_neuron_idxs=exc_neuron_idxs,
            inh_neuron_idxs=inh_neuron_idxs,
            center_input=center_input # true for EM
        )
        dataloaders_left = get_LSV1M_dataloaders_preprocessed(
            population=population,
            batch_size=batch_size,
            n_images_val=n_images_val,
            exc_neuron_idxs=exc_neuron_idxs_left,
            inh_neuron_idxs=inh_neuron_idxs_left,
            center_input=center_input
        )
    return dataloaders, dataloaders_left
  

def check_duplicate_keys(config_dicts): 
    all_keys = {}
    for config_name, config_dict in config_dicts:
        for key in config_dict.keys():
            if key in all_keys:
                print(f"Warning: Duplicate key '{key}' found in {config_name} and {all_keys[key]}")
            else:
                all_keys[key] = config_name

# brcnn_specific function
def set_brcnn_positions(model, neuron_idxs, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl'):
    """Set positions and orientations for BRCNN model."""
    pos1, pos2, ori = get_positions_and_orientations(pos_ori_path, neuron_idxs)
    
    readout = model.readout['all_sessions'].mu.data
    readout[0, 0, :, 0, 0] = pos1 / 5.5
    readout[0, 0, :, 0, 1] = -pos2 / 5.5
    readout[0, 0, :, 0, 2] = ori / np.pi 
