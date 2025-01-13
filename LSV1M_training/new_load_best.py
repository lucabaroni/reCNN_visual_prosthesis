#%%
import torch
import numpy as np
import random
from models import BRCNN_no_scaling, EnergyModel
from pickle_utils import pickleread
from dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders

def set_random_seeds(seed=1):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_neuron_indices(neurons_subset):
    """Get excitatory and inhibitory neuron indices based on subset.
    
    Args:
        neurons_subset (str): Which subset ('0', '1', or 'both')
        
    Returns:
        tuple: (excitatory indices, inhibitory indices)
    """
    neuron_ranges = {
        '0': (np.arange(20000), np.arange(5000)),
        '1': (np.arange(20000, 37500), np.arange(5000, 9375)), 
        'both': (np.arange(37500), np.arange(9375))
    }
    
    if neurons_subset not in neuron_ranges:
        raise ValueError("neurons_subset must be '0', '1', or 'both'")
        
    return neuron_ranges[neurons_subset]

def get_population_indices(population, exc_neuron_idxs, inh_neuron_idxs):
    """Get combined neuron indices based on population type.
    
    Args:
        population (str): Population type ('both', 'exc', or 'inh')
        exc_neuron_idxs (array): Excitatory neuron indices
        inh_neuron_idxs (array): Inhibitory neuron indices
        
    Returns:
        list: Combined neuron indices
    """
    if population == 'both':
        return list(exc_neuron_idxs) + list(inh_neuron_idxs + 37500)
    elif population == 'exc':
        return list(exc_neuron_idxs)
    elif population == 'inh':
        return list(inh_neuron_idxs + 37500)
    else:
        raise ValueError("population must be 'both', 'exc', or 'inh'")

def get_positions_and_orientations(pos_ori_path, neuron_idxs):
    """Load and process positions and orientations for neurons.
    
    Args:
        pos_ori_path (str): Path to positions and orientations file
        neuron_idxs (array-like): Indices of neurons to get data for
        
    Returns:
        tuple: (positions_x, positions_y, orientations)
    """
    pos_ori = pickleread(pos_ori_path)
    
    # Concatenate excitatory and inhibitory data
    pos_x = np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']])
    pos_y = np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']])
    ori = np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']])
    
    # Convert to tensors and select neurons
    pos_x = torch.Tensor(pos_x)[neuron_idxs]
    pos_y = torch.Tensor(pos_y)[neuron_idxs]
    ori = torch.Tensor(ori)[neuron_idxs]
        
    return pos_x, pos_y, ori

def load_brcnn_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
    """Load and configure a pretrained BRCNN model."""
    
    set_random_seeds()

    # Load saved model state
    d = torch.load('/project/LSV1M_training/artifacts/core_sd:v8/core_sd_and_config.pth')
    core_state_dict = d['core_state_dict']
    config = d['config']

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

    # Model configuration
    model_config = dict(
        num_rotations=32,
        stride=1,
        input_regularizer='LaplaceL2norm',
        upsampling=config.get('upsampling', 1),
        rot_eq_batch_norm=config.get('rot_eq_batch_norm', True),
        hidden_channels=config['hidden_channels'],
        hidden_kern=config['hidden_kern'],
        input_kern=config.get('input_kern', 3),
        layers=config['layers'],
        gamma_hidden=0 if config['gamma_hidden'] == 0 else config['gamma_hidden_nonzero'],
        gamma_input=0 if config['gamma_input'] == 0 else config['gamma_input_nonzero'],
        depth_separable=config['depth_separable'],
        init_sigma_range=config['init_sigma_range'],
        use_avg_reg=False,
    )

    # Initialize and load model
    model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    model.cuda()
    model.load_state_dict(core_state_dict, strict=False)

    # Get positions and orientations
    pos1, pos2, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', 
                                                    neuron_idxs)

    # Set positions for selected neurons
    readout = model.readout['all_sessions'].mu.data
    readout[0, 0, :, 0, 0] = pos1 / 5.5
    readout[0, 0, :, 0, 1] = -pos2 / 5.5
    readout[0, 0, :, 0, 2] = ori / np.pi
    
    model.eval()
    return (model, dataloaders) if return_dataloaders else model

def load_best_energy_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
    """Load the best energy model checkpoint and configure it."""
    
    # Load checkpoint
    checkpoint = torch.load('dummy_path/model_checkpoint.pth') #TODO FIX PATH
    model_config = checkpoint['config']
    model_state_dict = checkpoint['model_state_dict']

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

    # Initialize and load model
    model = EnergyModel(dataloaders, seed=0, **model_config)
    model.cuda()
    model.load_state_dict(model_state_dict)

    # Get positions and orientations
    pos1, pos2, ori = get_positions_and_orientations('/CSNG/baroni/Dic23data/pos_and_ori.pkl', 
                                                    neuron_idxs)

    # Update model config
    model_config.update({
        'positions_x': -pos1, 
        'positions_y': pos2,
        'orientations': -ori
    })
    
    model.eval()
    return (model, dataloaders) if return_dataloaders else model


#%%
#%% old code
# import torch
# import numpy as np
# import random
# from models import BRCNN_no_scaling, EnergyModel
# from pickle_utils import pickleread
# from dataset import get_LSV1M_dataloaders, get_LSV1M_empty_dataloaders

# def load_brcnn_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
#     """
#     Load and configure a pretrained BRCNN model.
    
#     Args:
#         population (str): Which neuron population to use ('both', 'exc', or 'inh')
#         dataloaders_batch_size (int): Batch size for dataloaders
#         dataloaders_n_images_val (int): Number of validation images 
#         neurons_subset (str): Which subset of neurons to use ('0', '1', or 'both')
#         return_dataloaders (bool): Whether to return dataloaders along with model
        
#     Returns:
#         model: Configured PyTorch model
#         dataloaders: Dictionary of train/val dataloaders (if return_dataloaders=True)
#     """
#     # Set random seeds for reproducibility
#     seed = 1
#     torch.manual_seed(seed)
#     np.random.seed(seed) 
#     random.seed(seed)

#     # Load saved model state
#     d = torch.load('/project/LSV1M_training/artifacts/core_sd:v8/core_sd_and_config.pth')
#     core_state_dict = d['core_state_dict']
#     config = d['config']

#     # Configure neuron indices based on subset
#     neuron_ranges = {
#         '0': (np.arange(20000), np.arange(5000)),
#         '1': (np.arange(20000, 37500), np.arange(5000, 9375)),
#         'both': (np.arange(37500), np.arange(9375))
#     }
    
#     if neurons_subset not in neuron_ranges:
#         raise ValueError("neurons_subset must be '0', '1', or 'both'")
        
#     exc_neuron_idxs, inh_neuron_idxs = neuron_ranges[neurons_subset]

#     # Select population
#     if population == 'both':
#         neuron_idxs = list(exc_neuron_idxs) + list(inh_neuron_idxs + 37500)
#     elif population == 'exc':
#         neuron_idxs = list(exc_neuron_idxs)
#     elif population == 'inh':
#         neuron_idxs = list(inh_neuron_idxs + 37500)
#     else:
#         raise ValueError("population must be 'both', 'exc', or 'inh'")

#     # Get appropriate dataloaders
#     dataloader_fn = get_LSV1M_dataloaders if return_dataloaders else get_LSV1M_empty_dataloaders
#     dataloaders = dataloader_fn(
#         population=population,
#         n_images_val=dataloaders_n_images_val,
#         exc_neuron_idxs=exc_neuron_idxs,
#         inh_neuron_idxs=inh_neuron_idxs,
#         batch_size=dataloaders_batch_size,
#         normalize_input=True,
#         center_input=False,
#         normalize_target=True if return_dataloaders else None
#     )

#     # Model configuration
#     model_config = dict(
#         num_rotations=32,
#         stride=1,
#         input_regularizer='LaplaceL2norm',
#         upsampling=config.get('upsampling', 1),
#         rot_eq_batch_norm=config.get('rot_eq_batch_norm', True),
#         hidden_channels=config['hidden_channels'],
#         hidden_kern=config['hidden_kern'],
#         input_kern=config.get('input_kern', 3),
#         layers=config['layers'],
#         gamma_hidden=0 if config['gamma_hidden'] == 0 else config['gamma_hidden_nonzero'],
#         gamma_input=0 if config['gamma_input'] == 0 else config['gamma_input_nonzero'],
#         depth_separable=config['depth_separable'],
#         init_sigma_range=config['init_sigma_range'],
#         use_avg_reg=False,
#     )

#     # Initialize and load model
#     model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
#     model.cuda()
#     model.load_state_dict(core_state_dict, strict=False)

#     # Configure readout positions
#     pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
    
#     # Load all positions and orientations
#     all_pos1 = torch.Tensor(np.concatenate([
#         pos_ori['V1_Exc_L23']['pos_x'], 
#         pos_ori['V1_Inh_L23']['pos_x']
#     ])) / 5.5
    
#     all_pos2 = torch.Tensor(np.concatenate([
#         pos_ori['V1_Exc_L23']['pos_y'],
#         pos_ori['V1_Inh_L23']['pos_y']
#     ])) / 5.5
    
#     all_ori = torch.Tensor(np.concatenate([
#         pos_ori['V1_Exc_L23']['ori'],
#         pos_ori['V1_Inh_L23']['ori']
#     ])) / np.pi

#     # Set positions for selected neurons
#     readout = model.readout['all_sessions'].mu.data
#     readout[0, 0, :, 0, 0] = all_pos1[neuron_idxs]
#     readout[0, 0, :, 0, 1] = -all_pos2[neuron_idxs]
#     readout[0, 0, :, 0, 2] = all_ori[neuron_idxs]
    
#     model.eval()
#     return (model, dataloaders) if return_dataloaders else model


# def load_best_energy_model(population='both', dataloaders_batch_size=300, dataloaders_n_images_val=5000, neurons_subset='0', return_dataloaders=False):
#     """Load the best energy model checkpoint and configure it for the specified neurons.
    
#     Args:
#         population (str): Which population to use ('both', 'exc', or 'inh')
#         dataloaders_batch_size (int): Batch size for dataloaders
#         dataloaders_n_images_val (int): Number of validation images
#         neurons_subset (str): Which subset of neurons to use ('0', '1' or 'both')
#         return_dataloaders (bool): Whether to return dataloaders along with model
        
#     Returns:
#         model or (model, dataloaders)
#     """
#     # Load checkpoint with model config and state dict
#     checkpoint = torch.load('dummy_path/model_checkpoint.pth')
#     model_config = checkpoint['config']
#     model_state_dict = checkpoint['model_state_dict']

#     # Define neuron indices based on subset
#     if neurons_subset == '0':
#         exc_neuron_idxs = np.arange(0, 20000)
#         inh_neuron_idxs = np.arange(0, 5000)
#     elif neurons_subset == '1':
#         exc_neuron_idxs = np.arange(20000, 37500)
#         inh_neuron_idxs = np.arange(5000, 9375)
#     else:  # subset 'both'
#         exc_neuron_idxs = np.arange(0, 37500)
#         inh_neuron_idxs = np.arange(0, 9375)

#     # Get dataloaders
#     dataloaders = get_LSV1M_dataloaders(
#         population=population,
#         batch_size=dataloaders_batch_size,
#         n_images_val=dataloaders_n_images_val,
#         exc_neuron_idxs=exc_neuron_idxs,
#         center_input=True,
#         inh_neuron_idxs=inh_neuron_idxs
#     )

#     # Initialize model with loaded config
#     model = EnergyModel(dataloaders, seed=0, **model_config)
#     model.cuda()

#     # Load saved state dict
#     model.load_state_dict(model_state_dict)

#     # Update positions and orientations for selected neurons
#     neuron_idxs = np.concatenate([exc_neuron_idxs, inh_neuron_idxs + 37500])
#     pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
    
#     # Get positions and orientations for selected neurons
#     pos2 = torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']]))[neuron_idxs]
#     pos1 = -torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']]))[neuron_idxs]
#     ori = -torch.Tensor(np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']]))[neuron_idxs]

#     # Update model config with new positions
#     model_config.update({
#         'positions_x': pos1,
#         'positions_y': pos2,
#         'orientations': ori
#     })
    
#     model.eval()
#     return (model, dataloaders) if return_dataloaders else model


