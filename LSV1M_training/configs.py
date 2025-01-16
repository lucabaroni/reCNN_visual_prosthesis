"""
Configuration settings for LSV1M training models.
Contains shared base configurations and model-specific settings for:
- Energy Model
- BRCNN Model
"""

def get_dataset_config(**kwargs):
    """Common dataset configuration settings"""
    return {
        'population': kwargs.get('population', 'both'),
        'batch_size': kwargs.get('batch_size', 16),  # BRCNN uses 16, EM uses 4
        'n_exc': kwargs.get('n_exc', 20000),
        'n_inh': kwargs.get('n_inh', 5000),
        'n_images_val': kwargs.get('n_images_val', 5000),
    }

def get_trainer_config(device='cuda', **kwargs):
    """Common trainer configuration settings"""
    return {
        'stop_function': 'get_correlations',
        'loss_function': 'MSE',
        'maximize': True,
        'avg_loss': False,
        'device': device,
        'max_iter': 50,  # BRCNN uses 1000, EM uses 10
        'lr_init': kwargs.get('lr_init', 0.001),
        'lr_decay_steps': kwargs.get('lr_decay_steps', 2),
        'patience': kwargs.get('patience', 2),
        'adamw': kwargs.get('adamw', False),
        'track_training': False, 
        'verbose': True,
    }

def get_energy_model_config(pos1, pos2, ori):
    """Energy Model specific configuration"""
    return {
        'positions_x': -pos2,
        'positions_y': pos1,
        'orientations': -ori,
        'nonlinearity': 'square_root',
        # 'pnl_vmin': -10,
        # 'pnl_vmax': 20,
        # 'pnl_nbis': 100,
        # 'pnl_smooth_reg_weight': 0,
        # 'pnl_smoothness_reg_order': 2,
        'resolution': [55, 55],
        'xlim': (-5.5, 5.5),
        'ylim': (-5.5, 5.5),
        'sigma_x_init': .4,
        'sigma_y_init': .4,
        'f_init': 0.7,
        'filter_scale_init': 0.01,
        'final_scale_init': 1,
        'keep_fixed_pos_and_ori': True,
        'data_info': None,
        'common_rescaling': True
    }

def get_brcnn_model_config(**kwargs):
    """BRCNN Model specific configuration"""
    return {
        'num_rotations': kwargs.get('num_rotations', 32),
        'stride': kwargs.get('stride', 1),
        'layers': kwargs.get('layers', 5),
        'hidden_channels': kwargs.get('hidden_channels', 8),
        'input_kern': kwargs.get('input_kern', 5),
        'hidden_kern': kwargs.get('hidden_kern', 5),
        'input_regularizer': kwargs.get('input_regularizer', 'LaplaceL2norm'),
        'gamma_hidden': kwargs.get('gamma_hidden', 0),
        'gamma_input': kwargs.get('gamma_input', 0),
        'use_avg_reg': kwargs.get('use_avg_reg', False),
        'upsampling': kwargs.get('upsampling', 1),
        'rot_eq_batch_norm': kwargs.get('rot_eq_batch_norm', True),
        'depth_separable': kwargs.get('depth_separable', True),
        'init_sigma_range': kwargs.get('init_sigma_range', 0.1),
        'do_not_sample': True,
        'freeze_positions_and_orientations': True   
    }



