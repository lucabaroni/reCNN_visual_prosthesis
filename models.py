
from mimetypes import common_types
# from predict_neural_responses.custom_layers import gauss
# from predict_neural_responses.models import *
from utils import get_fraction_oracles
# import predict_neural_responses.dnn_blocks.dnn_blocks as bl

from torch import nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

from neuralpredictors.measures.np_functions import corr as corr_from_neuralpredictors
from neuralpredictors.measures.np_functions import (
    oracle_corr_jackknife,
    oracle_corr_conservative,
)
from neuralpredictors.layers.cores.conv2d import RotationEquivariant2dCore

import logging

from readout import Gaussian3dCyclic, Gaussian3dCyclicNoScale
from core import RotationEquivariant2dCoreBottleneck
import matplotlib.pyplot as plt
from experiments.utils import reconstruct_orientation_maps, get_neuron_estimates

from neuralpredictors.layers.readouts.multi_readout import MultiReadoutBase

from readout import Gaussian3dCyclicNoScale
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
import numpy as np
from neuralpredictors.layers.activations import PiecewiseLinearExpNonlinearity

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0
        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)

        output_attn_weights = kwargs.get("output_attn_weights", False)
        if output_attn_weights:
            if len(x) == 2:
                x, attention_weights = x
            else:
                x, weights_1, weights_2 = x
                attention_weights = (weights_1, weights_2)
            return F.softplus(x) + 1, attention_weights
        return F.softplus(x)

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

class Encoder_2(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
        n_neurons = self.readout['all_sessions'].mu.shape[2]
        self.final_scale = nn.Parameter(torch.ones([1, n_neurons]))
        self.final_bias = nn.Parameter(torch.zeros([1, n_neurons]))

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0
        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)

        output_attn_weights = kwargs.get("output_attn_weights", False)
        if output_attn_weights:
            if len(x) == 2:
                x, attention_weights = x
            else:
                x, weights_1, weights_2 = x
                attention_weights = (weights_1, weights_2)
            return F.softplus(x) + 1, attention_weights
        return self.final_scale * F.softplus(x) + self.final_bias

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

class Encoder_common_scaling(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout
        n_neurons = self.readout['all_sessions'].mu.shape[2]
        self.final_scale = nn.Parameter(torch.ones([1, 1]))
        self.final_bias = nn.Parameter(torch.zeros([1, 1]))

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0
        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)

        output_attn_weights = kwargs.get("output_attn_weights", False)
        if output_attn_weights:
            if len(x) == 2:
                x, attention_weights = x
            else:
                x, weights_1, weights_2 = x
                attention_weights = (weights_1, weights_2)
            return F.softplus(x) + 1, attention_weights
        return self.final_scale * F.softplus(x) + self.final_bias

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)


class Encoder_no_scale(nn.Module):
    def __init__(self, core, readout):
        super().__init__()
        self.core = core
        self.readout = readout

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        if repeat_channel_dim is not None:
            x = x.repeat(1, repeat_channel_dim, 1, 1)
            x[:, 1:, ...] = 0
        x = self.core(x)
        x = self.readout(x, data_key=data_key, **kwargs)
        return x 

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

def unpack_data_info(data_info):
    in_shapes_dict = {k: v["input_dimensions"] for k, v in data_info.items()}
    input_channels = [v["input_channels"] for k, v in data_info.items()]
    n_neurons_dict = {k: v["output_dimension"] for k, v in data_info.items()}
    return n_neurons_dict, in_shapes_dict, input_channels

def reCNN_bottleneck_MultiCyclicGauss3d_with_final_scaling(
    dataloaders,
    seed,
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    core_hidden_channels = 16, 
    core_hidden_kern = 5, 
    core_input_kern = 7,
    core_layers = 5, 
    core_gamma_hidden = 0.01,
    core_gamma_input =  0.1,
    depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,

):
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)
    stack = -1

    core = RotationEquivariant2dCoreBottleneck(
        input_channels=core_input_channels,
        num_rotations=num_rotations,
        stride=stride,
        upsampling=upsampling,
        rot_eq_batch_norm=rot_eq_batch_norm,
        input_regularizer=input_regularizer,
        # input_channels=input_channels,
        hidden_channels=core_hidden_channels,
        input_kern=core_input_kern,
        hidden_kern=core_hidden_kern,
        layers=core_layers,
        gamma_input=core_gamma_input,
        gamma_hidden=core_gamma_hidden,
        stack=stack,
        depth_separable=depth_separable,
        use_avg_reg=use_avg_reg,
        bottleneck_kernel=bottleneck_kernel
        )

    readout = MultiReadoutBase(
        loaders = dataloaders,
        in_shape_dict = {'all_sessions': [num_rotations, 46,46]},
        n_neurons_dict = n_neurons_dict, 
        base_readout=Gaussian3dCyclicNoScale, 
        mean_activity_dict=None, 
        # add here all other arguments
        bias=readout_bias, 
        init_mu_range=init_mu_range,
        init_sigma_range=init_sigma_range,
        gamma_readout=readout_gamma,
        core = core,
        )
    
    model = Encoder_2(core, readout)
    return model


def reCNN_bottleneck_MultiCyclicGauss3d_no_scaling(
    dataloaders,
    seed,
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    core_hidden_channels = 16, 
    core_hidden_kern = 5, 
    core_input_kern = 7,
    core_layers = 5, 
    core_gamma_hidden = 0.01,
    core_gamma_input =  0.1,
    depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,

):
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )

    set_random_seed(seed)
    stack = -1

    core = RotationEquivariant2dCoreBottleneck(
        input_channels=core_input_channels,
        num_rotations=num_rotations,
        stride=stride,
        upsampling=upsampling,
        rot_eq_batch_norm=rot_eq_batch_norm,
        input_regularizer=input_regularizer,
        # input_channels=input_channels,
        hidden_channels=[core_hidden_channels]*core_layers,
        input_kern=core_input_kern,
        hidden_kern=core_hidden_kern,
        layers=core_layers,
        gamma_input=core_gamma_input,
        gamma_hidden=core_gamma_hidden,
        stack=stack,
        depth_separable=depth_separable,
        use_avg_reg=use_avg_reg,
        bottleneck_kernel=bottleneck_kernel
        )

    readout = MultiReadoutBase(
        loaders = dataloaders,
        in_shape_dict = {'all_sessions': [num_rotations, 46,46]},
        n_neurons_dict = n_neurons_dict, 
        base_readout=Gaussian3dCyclicNoScale, 
        mean_activity_dict=None, 
        # add here all other arguments
        bias=readout_bias, 
        init_mu_range=init_mu_range,
        init_sigma_range=init_sigma_range,
        gamma_readout=readout_gamma,
        core = core,
        )
    
    model = Encoder(core, readout)
    return model


class EnergyModel(nn.Module):
    def __init__(self,
        dataloaders,
        seed,
        positions_x, 
        positions_y,
        orientations,
        nonlinearity='square_root',
        pnl_vmin = -10, 
        pnl_vmax = 10, 
        pnl_nbis = 300, 
        pnl_smooth_reg_weight=1,
        pnl_smoothness_reg_order=2,
        resolution = [46,46], 
        xlim = (-2.67, 2.67), 
        ylim = (-2.67, 2.67), 
        sigma_x_init=0.2, 
        sigma_y_init=0.2,
        f_init = 1, 
        filter_scale_init = 1, 
        final_scale_init = 1, 
        keep_fixed_pos_and_ori = False,
        data_info=None,
        common_rescaling=False):
        super().__init__()
        set_random_seed(seed)
        if data_info is not None:
            n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
        else:
            if "train" in dataloaders.keys():
                dataloaders = dataloaders["train"]
            # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
            in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]
            session_shape_dict = get_dims_for_loader_dict(dataloaders)
            n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
            in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
            input_channels = [v[in_name][1] for v in session_shape_dict.values()]
        print('n_neurons_dict', n_neurons_dict)
        print('in_shapes_dict', in_shapes_dict)
        print('input_channels', input_channels)

        self.num_neurons = n_neurons_dict['all_sessions']
        self.sigma_x = torch.nn.Parameter(torch.ones(1) * sigma_x_init)
        self.sigma_y = torch.nn.Parameter(torch.ones(1) * sigma_y_init)
        self.f = torch.nn.Parameter(torch.ones(1) * f_init)

        self.filter_scale = torch.nn.Parameter(torch.ones(1) * filter_scale_init)
        self.filter_bias = torch.nn.Parameter(torch.zeros(1))

        if common_rescaling==True:
            self.final_bias = torch.nn.Parameter(torch.ones(1))
            self.final_scale = torch.nn.Parameter(torch.ones(1) * final_scale_init)
        else:
            self.final_bias = torch.nn.Parameter(torch.ones(1, self.num_neurons))
            self.final_scale = torch.nn.Parameter(torch.ones(1, self.num_neurons) * final_scale_init)

        if keep_fixed_pos_and_ori:
            self.register_buffer("positions_x", torch.Tensor(positions_x))
            self.register_buffer("positions_y", torch.Tensor(positions_y))
            self.register_buffer("orientations", torch.Tensor(orientations))
        else: 
            self.positions_x = nn.Parameter(torch.Tensor(positions_x))
            self.positions_y = nn.Parameter(torch.Tensor(positions_y))
            self.orientations = nn.Parameter(torch.Tensor(orientations))

        self.init_meshgrids(resolution, xlim, ylim)
        self.nonlinearity_name = nonlinearity
        if nonlinearity=='square_root':
            self.nonlinearity = torch.sqrt
        if nonlinearity=='piecewise_nonlinearity':
            self.nonlinearity = PiecewiseLinearExpNonlinearity(
                1,
                bias=False, 
                vmin=pnl_vmin, 
                vmax=pnl_vmax, 
                num_bins=pnl_nbis, 
                smooth_reg_weight=pnl_smooth_reg_weight, 
                smoothnes_reg_order=pnl_smoothness_reg_order)

    def init_meshgrids(self, resolution, ylim, xlim):
        x = torch.linspace(xlim[0], xlim[1], resolution[0])
        y = torch.linspace(ylim[0], ylim[1], resolution[1])
        meshgrid_x, meshgrid_y= torch.meshgrid(x, y)
        meshgrid_x = meshgrid_x.reshape(1, *meshgrid_x.shape).repeat(self.num_neurons, 1,1)
        meshgrid_y = meshgrid_y.reshape(1, *meshgrid_y.shape).repeat(self.num_neurons, 1,1)
        self.register_buffer('meshgrid_x', meshgrid_x)
        self.register_buffer('meshgrid_y', meshgrid_y)
    
    def get_rotated_meshgrids(self):
        meshgrid_x_rotated = self.meshgrid_x * torch.cos(self.orientations[:, None, None]) - self.meshgrid_y * torch.sin(self.orientations[:, None, None])
        meshgrid_y_rotated = self.meshgrid_x * torch.sin(self.orientations[:, None, None]) + self.meshgrid_y * torch.cos(self.orientations[:, None, None])
        return meshgrid_x_rotated, meshgrid_y_rotated

    def get_rotated_pos(self):
        pos_x_rotated = torch.cos(self.orientations) * self.positions_x - torch.sin(self.orientations) * self.positions_y
        pos_y_rotated = torch.sin(self.orientations) * self.positions_x + torch.cos(self.orientations) * self.positions_y
        return pos_x_rotated, pos_y_rotated

    def get_modulatory_gaussians(self, meshgrid_x_rotated, meshgrid_y_rotated, pos_x_rotated, pos_y_rotated):
        modulatory_gaussian = torch.exp(
            -0.5*(
                (torch.square(meshgrid_x_rotated - pos_x_rotated[:, None, None])/(torch.square(self.sigma_x) + 0.000005))
                + (torch.square(meshgrid_y_rotated - pos_y_rotated[:, None, None]) /(torch.square(self.sigma_y) + 0.000005)))
                )
        return modulatory_gaussian

    def get_cosine_waves(self, meshgrid_x_rotated, pos_x_rotated):
        cos_wave_even = torch.cos(2*np.pi*self.f*(meshgrid_x_rotated + pos_x_rotated[:, None, None])) 
        cos_wave_odd = torch.cos(2*np.pi*self.f*(meshgrid_x_rotated + pos_x_rotated[:, None, None]) + np.pi/2) 
        return cos_wave_even, cos_wave_odd

    def get_filters_even_and_odd(self):
        meshgrid_x_rotated, meshgrid_y_rotated = self.get_rotated_meshgrids()
        pos_x_rotated, pos_y_rotated = self.get_rotated_pos()
        modulatory_gaussian = self.get_modulatory_gaussians(meshgrid_x_rotated, meshgrid_y_rotated, pos_x_rotated, pos_y_rotated)
        cos_wave_even, cos_wave_odd = self.get_cosine_waves(meshgrid_x_rotated, pos_x_rotated)
        f_even = modulatory_gaussian*cos_wave_even
        f_odd = modulatory_gaussian*cos_wave_odd
        return f_even, f_odd

    def forward(self, x, data_key=None, repeat_channel_dim=None, **kwargs):
        f_even, f_odd = self.get_filters_even_and_odd()
        f_even = self.filter_scale*f_even
        f_odd = self.filter_scale*f_odd
        x_even = torch.einsum('bixy, nxy -> bn', x, f_even)
        x_odd = torch.einsum('bixy, nxy -> bn', x, f_odd)
        x = torch.square(x_even) + torch.square(x_odd)
        if self.nonlinearity_name=='piecewise_nonlinearity':
            x = x - self.filter_bias
        x_shape = x.shape
        x = self.nonlinearity(x.reshape(-1, 1)).reshape(x_shape)
        x = self.final_scale * x + self.final_bias
        return x

    def regularizer(self, data_key=None, **kwargs):
        return 0

def BRCNN_no_scaling(
    dataloaders,
    seed,
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    hidden_channels = 16, 
    hidden_kern = 5, 
    input_kern = 7,
    layers = 5, 
    gamma_hidden = 0.01,
    gamma_input =  0.1,
    depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    do_not_sample=True,
    freeze_positions_and_orientations=True,
    data_info=None,
):
    set_random_seed(seed)
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]
        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]
        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )
    readout_in_shape_dict = { k: [num_rotations, *v[-2:]] for k,v in in_shapes_dict.items()}
    stack = -1

    core = RotationEquivariant2dCore(
        input_channels=core_input_channels, 
        num_rotations=num_rotations,
        stride=stride, 
        upsampling=upsampling,
        rot_eq_batch_norm=rot_eq_batch_norm,
        input_regularizer=input_regularizer, 
        input_kern=input_kern,
        hidden_kern = hidden_kern,
        hidden_channels=[hidden_channels]*(layers-1) + [1],
        layers = layers,
        gamma_input=gamma_input,
        gamma_hidden=gamma_hidden,
        stack = stack, 
        depth_separable=depth_separable,
        use_avg_reg=use_avg_reg
        )

    readout = MultiReadoutBase(
        loaders = dataloaders,
        in_shape_dict = readout_in_shape_dict,
        n_neurons_dict = n_neurons_dict, 
        base_readout=Gaussian3dCyclicNoScale, 
        mean_activity_dict=None, 
        bias=readout_bias, 
        init_mu_range=init_mu_range,
        init_sigma_range=init_sigma_range,
        gamma_readout=readout_gamma,
        core = core,
        # parameters to freeze pos and ori 
        do_not_sample=do_not_sample,
        freeze_positions_and_orientations=freeze_positions_and_orientations,
        )
    
    model = Encoder_no_scale(core, readout)
    return model



def BRCNN_with_final_scaling(
    dataloaders,
    seed,
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    hidden_channels = 16, 
    hidden_kern = 5, 
    input_kern = 7,
    layers = 5, 
    gamma_hidden = 0.01,
    gamma_input =  0.1,
    depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,
):
    set_random_seed(seed)
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]
        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]
        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )
    readout_in_shape_dict = { k: [num_rotations, *v[-2:]] for k,v in in_shapes_dict.items()}
    stack = -1

    core = RotationEquivariant2dCore(
        input_channels=core_input_channels, 
        num_rotations=num_rotations,
        stride=stride, 
        upsampling=upsampling,
        rot_eq_batch_norm=rot_eq_batch_norm,
        input_regularizer=input_regularizer, 
        input_kern=input_kern,
        hidden_kern = hidden_kern,
        hidden_channels=[hidden_channels]*(layers-1) + [1],
        layers = layers,
        gamma_input=gamma_input,
        gamma_hidden=gamma_hidden,
        stack = stack, 
        depth_separable=depth_separable,
        use_avg_reg=use_avg_reg)

    readout = MultiReadoutBase(
        loaders = dataloaders,
        in_shape_dict = readout_in_shape_dict,
        n_neurons_dict = n_neurons_dict, 
        base_readout=Gaussian3dCyclicNoScale, 
        mean_activity_dict=None, 
        bias=readout_bias, 
        init_mu_range=init_mu_range,
        init_sigma_range=init_sigma_range,
        gamma_readout=readout_gamma,
        core = core,
        )
    
    model = Encoder_2(core, readout)
    return model


def BRCNN_with_common_scaling(
    dataloaders,
    seed,
    num_rotations=32, 
    upsampling=2, 
    stride=1, 
    rot_eq_batch_norm=True, 
    input_regularizer='LaplaceL2norm',
    hidden_channels = 16, 
    hidden_kern = 5, 
    input_kern = 7,
    layers = 5, 
    gamma_hidden = 0.01,
    gamma_input =  0.1,
    depth_separable = True,
    use_avg_reg=False, 
    bottleneck_kernel=5,
    #readout
    readout_bias=False, 
    readout_gamma=0,
    init_mu_range=0.1, 
    init_sigma_range=0.1,
    data_info=None,
):
    set_random_seed(seed)
    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]
        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]
        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    core_input_channels = (
        list(input_channels.values())[0]
        if isinstance(input_channels, dict)
        else input_channels[0]
    )
    readout_in_shape_dict = { k: [num_rotations, *v[-2:]] for k,v in in_shapes_dict.items()}
    stack = -1

    core = RotationEquivariant2dCore(
        input_channels=core_input_channels, 
        num_rotations=num_rotations,
        stride=stride, 
        upsampling=upsampling,
        rot_eq_batch_norm=rot_eq_batch_norm,
        input_regularizer=input_regularizer, 
        input_kern=input_kern,
        hidden_kern = hidden_kern,
        hidden_channels=[hidden_channels]*(layers-1) + [1],
        layers = layers,
        gamma_input=gamma_input,
        gamma_hidden=gamma_hidden,
        stack = stack, 
        depth_separable=depth_separable,
        use_avg_reg=use_avg_reg)

    readout = MultiReadoutBase(
        loaders = dataloaders,
        in_shape_dict = readout_in_shape_dict,
        n_neurons_dict = n_neurons_dict, 
        base_readout=Gaussian3dCyclicNoScale, 
        mean_activity_dict=None, 
        bias=readout_bias, 
        init_mu_range=init_mu_range,
        init_sigma_range=init_sigma_range,
        gamma_readout=readout_gamma,
        core = core,
        train_pos=True,
        )
    
    model = Encoder_common_scaling(core, readout)
    return model