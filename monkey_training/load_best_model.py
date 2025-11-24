# %% LOAD THE BEST MODEL
from pickle_utils import pickleread
from nnfabrik.builder import get_data
from models import BRCNN_no_scaling, EnergyModel
import torch
import numpy as np
import h5py
import pandas as pd

def load_model_checkpoint(checkpoint_path):
    """Load model checkpoint and return state dict and config"""
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['model_state_dict'], checkpoint['model_config']


def get_subject_filenames(filenames):
    """Split filenames by subject and get indices. Works only on the old data format."""
    subject_0_files = []
    subject_1_files = []
    ids0 = []
    ids1 = []
    skip = 0
    
    for file in filenames:
        n = len(eval(pd.read_csv(file + '/responses.csv')['responses'][0]))
        subject_id = pd.read_json(file + '/meta_data.json')['subject_id'].iloc[0]
        if subject_id == 31:
            subject_0_files.append(file)
            ids0.extend(list(np.arange(skip, skip + n)))
        else:
            subject_1_files.append(file)
            ids1.extend(list(np.arange(skip, skip + n)))
        skip += n
    return subject_0_files, subject_1_files, ids0, ids1



def get_subject_filenames_old(filenames):
    """Split filenames by subject and get indices. Works only on the old data format."""
    subject_0_files = []
    subject_1_files = []
    ids0 = []
    ids1 = []
    skip = 0
    
    for file in filenames:
        n = len(pickleread(file)['training_responses'])
        if pickleread(file)['subject_id'] == 31:
            subject_0_files.append(file)
            ids0.extend(list(np.arange(skip, skip + n)))
        else:
            subject_1_files.append(file)
            ids1.extend(list(np.arange(skip, skip + n)))
        skip += n
        
    return subject_0_files, subject_1_files, ids0, ids1

def get_dataset_config(neuronal_data_files, batch_size=4):
    """Get dataset configuration"""
    return ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined_new_data_format',
            {'dataset': 'CSRF19_V1',
             'neuronal_data_files': neuronal_data_files,
             'image_cache_path': '../v1_data/images_npy/',
             'crop': 70,
             'subsample': 1,
             'seed': 1000,
             'scale': 0.5,
             'time_bins_sum': 12,
             'batch_size': batch_size,
             'normalize_resps': True})

def get_dataset_config_old(neuronal_data_files, batch_size=4):
    """Get dataset configuration"""
    return ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
            {'dataset': 'CSRF19_V1',
             'neuronal_data_files': neuronal_data_files,
             'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
             'crop': 70,
             'subsample': 1,
             'seed': 1000,
             'scale': 0.5,
             'time_bins_sum': 12,
             'batch_size': batch_size,
             'normalize_resps': True})

def load_neuron_preferences(h5_path, num_neurons=458):
    """Load neuron preferences from h5 file"""
    data = {i: {} for i in range(num_neurons)}
    with h5py.File(h5_path, 'r') as f:
        for i in range(num_neurons):
            data[i]['preferred_ori'] = f[f'full_field_params/neuron_{i}'][:][0]
            data[i]['preferred_pos'] = f[f'preferred_pos/neuron_{i}'][:]
            
    pos1, pos2, ori = [], [], []
    for i in range(num_neurons):
        pos1.append((data[i]['preferred_pos'][0]-(93/2))/(93/2))
        pos2.append((data[i]['preferred_pos'][1]-(93/2))/(93/2))
        ori.append(data[i]['preferred_ori']/np.pi)
        
    return map(torch.Tensor, (pos1, pos2, ori))

def get_monkey_subjects_info():
    """Get filenames and indices for both monkey subjects. Works only on the old data format."""
    all_filenames = [
        '../v1_data/train/3631896544452',
        '../v1_data/train/3632669014376',
        '../v1_data/train/3632932714885',
        '../v1_data/train/3633364677437',
        '../v1_data/train/3634055946316',
        '../v1_data/train/3634142311627',
        '../v1_data/train/3634658447291',
        '../v1_data/train/3634744023164',
        '../v1_data/train/3635178040531',
        '../v1_data/train/3635949043110',
        '../v1_data/train/3636034866307',
        '../v1_data/train/3636552742293',
        '../v1_data/train/3637161140869',
        '../v1_data/train/3637248451650',
        '../v1_data/train/3637333931598',
        '../v1_data/train/3637760318484',
        '../v1_data/train/3637851724731',
        '../v1_data/train/3638367026975',
        '../v1_data/train/3638456653849',
        '../v1_data/train/3638885582960',
        '../v1_data/train/3638373332053',
        '../v1_data/train/3638541006102',
        '../v1_data/train/3638802601378',
        '../v1_data/train/3638973674012',
        '../v1_data/train/3639060843972',
        '../v1_data/train/3639406161189',
        '../v1_data/train/3640011636703',
        '../v1_data/train/3639664527524',
        '../v1_data/train/3639492658943',
        '../v1_data/train/3639749909659',
        '../v1_data/train/3640095265572',
        '../v1_data/train/3631807112901'
    ]
    subject_0_filenames, subject_1_filenames, ids0, ids1 = get_subject_filenames(all_filenames)
    return all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1


def get_monkey_subjects_info_old():
    """Get filenames and indices for both monkey subjects. Works only on the old data format."""
    all_filenames = [
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
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
        '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle'
    ]
    subject_0_filenames, subject_1_filenames, ids0, ids1 = get_subject_filenames_old(all_filenames)
    return all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1

def get_best_brcnn_model(subject='both', return_dataloaders=False):
    """Load the best BRCNN model for specified subject(s).
    
    Args:
        subject (str or int): Which subject's data to use - 0, 1, or 'both'. Defaults to 'both'.
        return_dataloaders (bool): Whether to return the dataloaders along with the model. Defaults to False.
        
    Returns:
        BRCNN_no_scaling: Loaded model configured for specified subject(s)
        dict, optional: Dataloaders if return_dataloaders is True
    """
    model_state_dict, model_config = load_model_checkpoint(
        'artifacts/best_model:v30/fine_tune=core_best_model.pt'
    )
    all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1 = get_monkey_subjects_info()
    
    filenames_to_use = {
        0: subject_0_filenames,
        1: subject_1_filenames,
        'both': all_filenames
    }.get(subject)
    
    if filenames_to_use is None:
        raise ValueError(f"Invalid subject value: {subject}")

    dataset_fn, dataset_config = get_dataset_config(filenames_to_use, batch_size=4)
    dataloaders = get_data(dataset_fn, dataset_config)
    
    model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    
    if subject == 0:
        model.load_state_dict(model_state_dict)
    else:
        core_state_dict = {k:v for k,v in model_state_dict.items() if k.startswith('core')}
        model.load_state_dict(core_state_dict, strict=False)
        
        pos1, pos2, ori = load_neuron_preferences('/project/results_convnext_model.h5')
        if subject == 1:
            pos1, pos2, ori = pos1[ids1], pos2[ids1], ori[ids1]
            
        model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
        model.readout['all_sessions'].mu.data[0,0,:,0,1] = pos2
        model.readout['all_sessions'].mu.data[0,0,:,0,2] = ori

    if return_dataloaders:
        return model, dataloaders
    return model

def get_best_energy_model(subject='both', return_dataloaders=False):
    """Load the best Energy Model for specified subject.
    
    Args:
        subject (int or str): Which subject's data to use - 0, 1, or 'both'. Defaults to 0.
        return_dataloaders (bool): Whether to return the dataloaders along with the model. Defaults to False.
        
    Returns:
        EnergyModel: Loaded model configured for specified subject§
        dict, optional: Dataloaders if return_dataloaders is True
    """
    model_state_dict, model_config = load_model_checkpoint(
        '/project/monkey_training/energy_model.pt' # energy model V1
    )
    all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1 = get_monkey_subjects_info()
    
    filenames_to_use = {
        0: subject_0_filenames,
        1: subject_1_filenames,
        'both': all_filenames
    }.get(subject)
    
    if filenames_to_use is None:
        raise ValueError(f"Invalid subject value: {subject}")

    dataset_fn, dataset_config = get_dataset_config(filenames_to_use, batch_size=4)
    dataset_config.update({'center_inputs': True}) # center inputs for energy model
    dataloaders = get_data(dataset_fn, dataset_config)

    pos1, pos2, ori = load_neuron_preferences('/project/results_convnext_model.h5')
    if subject == 0:
        pos1, pos2, ori = pos1[ids0], pos2[ids0], ori[ids0]
    elif subject == 1:
        pos1, pos2, ori = pos1[ids1], pos2[ids1], ori[ids1]
    
    model_config.update({
        'positions_x': pos2,
        'positions_y': pos1,
        'orientations': -ori*np.pi 
    })

    model = EnergyModel(dataloaders, seed=0, **model_config)

    if subject == 0:
        model.load_state_dict(model_state_dict)
    else:
        core_state_dict = {k:v for k,v in model_state_dict.items() 
                          if not any(k.startswith(p) for p in ['positions', 'orientations', 'meshgrid'])}
        model.load_state_dict(core_state_dict, strict=False)

    if return_dataloaders:
        return model, dataloaders
    return model


#old untested code

# def get_best_brcnn_model(subject='both'):
#     # Load model checkpoint
#     model_state_dict, model_config = load_model_checkpoint(
#         '/project/monkey_training/artifacts/best_model:v4/fine_tune=core_best_model.pt'
#     )
#     # Get subject-specific filenames and indices
#     all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1 = get_monkey_subjects_info()
    
#     # Get dataset config and dataloaders based on subject
#     batch_size = 4
#     if subject == 0:
#         filenames_to_use = subject_0_filenames
#     elif subject == 1:
#         filenames_to_use = subject_1_filenames
#     elif subject == 'both':
#         filenames_to_use = all_filenames
#     else:
#         raise ValueError(f"Invalid subject value: {subject}")

#     dataset_fn, dataset_config = get_dataset_config(filenames_to_use, batch_size)
#     dataloaders = get_data(dataset_fn, dataset_config)
    
#     # Create and initialize model
#     model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
    
#     if subject == 0:
#         # Load full state dict for subject 0
#         model.load_state_dict(model_state_dict)
#     else:
#         # Load core state dict only for subject 1 or both
#         core_state_dict = {k:v for k,v in model_state_dict.items() if k.startswith('core')}
#         model.load_state_dict(core_state_dict, strict=False)
        
#         # Set neuron preferences for subject 1 or both
#         pos1, pos2, ori = load_neuron_preferences('/project/results_convnext_model.h5')
#         if subject == 1:
#             pos1, pos2, ori = pos1[ids1], pos2[ids1], ori[ids1]
            
#         model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
#         model.readout['all_sessions'].mu.data[0,0,:,0,1] = pos2
#         model.readout['all_sessions'].mu.data[0,0,:,0,2] = ori

#     return model

# # %%

# def get_best_energy_model(subject=0):
#     # load model checkpoint
#     model_state_dict, model_config = load_model_checkpoint(
#         '/project/monkey_training/artifacts/best_model:v4/energy_model.pt' # check this path
#     )

#     # Get subject-specific filenames and indices
#     all_filenames, subject_0_filenames, subject_1_filenames, ids0, ids1 = get_monkey_subjects_info()
    
#      # Get dataset config and dataloaders based on subject
#     batch_size = 4
#     if subject == 0:
#         filenames_to_use = subject_0_filenames
#     elif subject == 1:
#         filenames_to_use = subject_1_filenames
#     elif subject == 'both':
#         filenames_to_use = all_filenames
#     else:
#         raise ValueError(f"Invalid subject value: {subject}")

#     dataset_fn, dataset_config = get_dataset_config(filenames_to_use, batch_size)
#     dataloaders = get_data(dataset_fn, dataset_config)

        
#     # Update model config with subject-specific parameters
#     if subject == 0:
#         pos1, pos2, ori = pos1[ids0], pos2[ids0], ori[ids0] 
#     elif subject == 1:
#         pos1, pos2, ori = pos1[ids1], pos2[ids1], ori[ids1]
    
#     model_config.update({
#         'positions_x': pos1,
#         'positions_y': pos2,
#         'orientations': ori
#     })

#     # Create and initialize model
#     model = EnergyModel(dataloaders, seed=0, **model_config)

#     # load model
#     if subject == 0:
#         # Load full state dict for subject 0
#         model.load_state_dict(model_state_dict)
#     else:
#         # Load core state dict only for subject 1 or both
#         core_state_dict = {}
#         for k, v in model_state_dict.items():
#             if k.startswith('positions')!=True and k.startswith('orientations')!=True and  k.startswith('meshgrid')!=True:
#                 core_state_dict[k] = v
#         model.load_state_dict(core_state_dict, strict=False)
#     return model


###############################
###### old tested code#########
###############################

# def get_best_brcnn_model(subject='both'):
#     x = torch.load('/project/monkey_training/artifacts/best_model:v4/fine_tune=core_best_model.pt')

#     model_state_dict = x['model_state_dict']
#     model_config = x['model_config']

#     filenames = ['/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631896544452.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632669014376.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3632932714885.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3633364677437.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634055946316.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634142311627.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634658447291.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3634744023164.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635178040531.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3635949043110.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636034866307.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3636552742293.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637161140869.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637248451650.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637333931598.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637760318484.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3637851724731.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638367026975.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638456653849.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638885582960.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638373332053.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638541006102.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638802601378.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3638973674012.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639060843972.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639406161189.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640011636703.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639664527524.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639492658943.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3639749909659.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3640095265572.pickle',
#         '/data/monkey/toliaslab/CSRF19_V1/neuronal_data/CSRF19_V1_3631807112901.pickle']

#     subject_0_filenames = []
#     subject_1_filenames = []
#     ids0 = []
#     ids1 = []
#     skip = 0
#     batch_size=4

#     for file in filenames: 
#         n = len(pickleread(file)['training_responses']) 
#         if pickleread(file)['subject_id']== 31: 
#             subject_0_filenames.append(file)
#             ids0 = ids0 + list(np.arange(skip, skip + n))

#         else:
#             subject_1_filenames.append(file)
#             ids1 = ids1 + list(np.arange(skip, skip + n ))
#         skip += n

#     if subject==0:
#         dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
#             {'dataset': 'CSRF19_V1',
#             'neuronal_data_files': subject_0_filenames,
#             'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
#             'crop': 70,
#             'subsample': 1,
#             'seed': 1000,
#             'scale': 0.5,
#             'time_bins_sum': 12,
#             'batch_size': batch_size, 
#             'normalize_resps':True})
#         dataloaders = get_data(dataset_fn, dataset_config)

#         model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
#         model.load_state_dict(model_state_dict)
#     if subject=='both':
#         core_state_dict = {}
#         for k,v in model_state_dict.items():
#             if k.startswith('core'):
#                 core_state_dict[k]=v
#         dataset_fn, dataset_config = ('nnvision.datasets.monkey_loaders.monkey_static_loader_combined',
#             {'dataset': 'CSRF19_V1',
#             'neuronal_data_files': filenames,
#             'image_cache_path': '/data/monkey/toliaslab/CSRF19_V1/images/',
#             'crop': 70,
#             'subsample': 1,
#             'seed': 1000,
#             'scale': 0.5,
#             'time_bins_sum': 12,
#             'batch_size': batch_size, 
#             'normalize_resps':True})
#         dataloaders = get_data(dataset_fn, dataset_config)

#         model = BRCNN_no_scaling(dataloaders, seed=0, **model_config)
#         model.load_state_dict(core_state_dict, strict=False)

#         h5 = '/project/results_convnext_model.h5'
#         data1 = {i: {} for i in range(458)}
#         with h5py.File(h5, 'r') as f:
#             for i in range(458):
#                 dataset_name = f'full_field_params/neuron_{i}'
#                 data1[i]['preferred_ori'] = f[dataset_name][:][0]
#                 dataset_name = f'preferred_pos/neuron_{i}'
#                 data1[i]['preferred_pos'] = f[dataset_name][:]

#         pos1 = []
#         pos2 = []
#         ori = []
#         for i in range(458):
#             pos1.append((data1[i]['preferred_pos'][0]-(93/2))/(93/2))
#             pos2.append((data1[i]['preferred_pos'][1]-(93/2))/(93/2))
#             ori.append(data1[i]['preferred_ori']/np.pi)
#         pos1 = torch.Tensor(pos1)
#         pos2 = torch.Tensor(pos2)
#         ori = torch.Tensor(ori)

#         model.readout['all_sessions'].mu.data[0,0,:,0,0] = pos1
#         model.readout['all_sessions'].mu.data[0,0,:,0,1]  = pos2
#         model.readout['all_sessions'].mu.data[0,0,:,0,2]  = ori
#     return model

# %%
