#%%
import torch
import numpy as np
import random
from models import BRCNN_no_scaling, EnergyModel
from pickle_utils import pickleread
from LSV1M_training.dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders
from LSV1M_training.LSV1M_utils import (
    set_random_seeds, 
    get_neuron_indices, 
    get_population_indices,
    get_positions_and_orientations,
    set_brcnn_positions
)
d = torch.load('models_artifacts/macaque_data_model/model.pt')

#%%

def load_model_x(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False): 
    set_random_seeds()
    d = torch.load(
        # '/home/baroni/recnn/LSV1M_training/artifacts/core_sd:v8/core_sd_and_config.pth'
        'models_artifacts/macaque_data_model/model.pt'
                   )
    config = d['config']
    core_state_dict = d['core_state_dict']
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices(neurons_subset)
    neuron_idxs = get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs)

    # Set hyperparameters
    upsampling = config['upsampling'] if 'upsampling' in config else 1
    rot_eq_batch_norm = config['rot_eq_batch_norm'] if 'rot_eq_batch_norm' in config else True
    hidden_channels = config['hidden_channels']
    hidden_kern = config['hidden_kern']
    input_kern = config['input_kern'] if 'input_kern' in config else 3
    layers = config['layers']
    gamma_hidden = 0 if config['gamma_hidden'] == 0 else config['gamma_hidden_nonzero']
    gamma_input = 0 if config['gamma_input'] == 0 else config['gamma_input_nonzero']
    depth_separable = config['depth_separable']
    init_sigma_range = config['init_sigma_range']
    population = config['population'] if 'population' in config else 'both'

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
    )


    # Get dataloaders
    dataloader_fn = get_LSV1M_dataloaders if return_dataloaders else get_LSV1M_empty_dataloaders
    dataloaders = dataloader_fn(
        population=population,
        n_images_val=dataloaders_n_images_val,
        exc_neuron_idxs=exc_neuron_idxs,
        inh_neuron_idxs=inh_neuron_idxs,
        batch_size=dataloaders_batch_size,
        normalize_input=True,
        center_input=False,
        normalize_target=True
    )

    # Initialize and load model
    model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    model.cuda()
    model.load_state_dict(core_state_dict, strict=False)

    # Set positions using utility function
    set_brcnn_positions(model, neuron_idxs, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl')
    
    model.eval()
    return (model, dataloaders) if return_dataloaders else model



def load_best_brcnn_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
    """Load and configure a pretrained BRCNN model."""
    
    set_random_seeds()
    d = torch.load('/home/baroni/recnn/LSV1M_training/saved_models/brcnn__model_checkpoint_20250117_001846.pth')
    model_config = d['model_config']
    model_state_dict = d['model_state_dict']
    core_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith('core')}

    # Get neuron indices
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices(neurons_subset)
    neuron_idxs = get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs)

    # Get dataloaders
    dataloader_fn = get_LSV1M_dataloaders if return_dataloaders else get_LSV1M_empty_dataloaders
    dataloaders = dataloader_fn(
        population=population,
        n_images_val=dataloaders_n_images_val,
        exc_neuron_idxs=exc_neuron_idxs,
        inh_neuron_idxs=inh_neuron_idxs,
        batch_size=dataloaders_batch_size,
        normalize_input=True,
        center_input=False,
        normalize_target=True
    )

    # Initialize and load model
    model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    model.cuda()
    model.load_state_dict(core_state_dict, strict=False)

    # Set positions using utility function
    set_brcnn_positions(model, neuron_idxs, pos_ori_path='/CSNG/baroni/Dic23data/pos_and_ori.pkl')
    
    model.eval()
    return (model, dataloaders) if return_dataloaders else model



def load_best_energy_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
    """Load the best energy model checkpoint and configure it."""
    
    set_random_seeds()
    d = torch.load('/home/baroni/recnn/LSV1M_training/saved_models/em__model_checkpoint_20250117_135034.pth')
    model_config = d['model_config']
    model_state_dict = d['model_state_dict']
    ######## Load checkpoint ###########
    # checkpoint = torch.load('dummy_path/model_checkpoint.pth') #TODO FIX PATH
    # model_config = checkpoint['config']
    # model_state_dict = checkpoint['model_state_dict']

    # Get neuron indices
    exc_neuron_idxs, inh_neuron_idxs = get_neuron_indices(neurons_subset)
    neuron_idxs = get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs)
    # Get dataloaders
    dataloader_fn = get_LSV1M_dataloaders if return_dataloaders else get_LSV1M_empty_dataloaders
    dataloaders = dataloader_fn(
        population=population,
        batch_size=dataloaders_batch_size,
        n_images_val=dataloaders_n_images_val,
        exc_neuron_idxs=exc_neuron_idxs,
        center_input=True, #NOTE this is different from the BRCNN model
        inh_neuron_idxs=inh_neuron_idxs
    )

    # Get positions and orientations
    pos1, pos2, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', neuron_idxs)

    # Update model config
    model_config.update({
        'positions_x': -pos2, 
        'positions_y': pos1,
        'orientations': -ori
    })

    # Initialize and load model
    model = EnergyModel(dataloaders, seed=0, **model_config)
    model.cuda()
    if neurons_subset == '0':
        model.load_state_dict(model_state_dict)
    else:
        shared_params_state_dict = {k:v for k,v in model_state_dict.items() if k not in ['positions_x', 'positions_y', 'orientations', 'meshgrid_x', 'meshgrid_y']}
        model.load_state_dict(shared_params_state_dict, strict=False)

    model.eval()
    return (model, dataloaders) if return_dataloaders else model

