
from mimetypes import common_types
# from predict_neural_responses.custom_layers import gauss
# from predict_neural_responses.models import *
from utils import get_fraction_oracles
# import predict_neural_responses.dnn_blocks.dnn_blocks as bl

from torch import nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

from neuralpredictors.layers.hermite import (
    HermiteConv2D,
)
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

logger = logging.getLogger(__name__)


# class ExtendedEncodingModel(encoding_model):
#     """Parent class for system identification enconding models, keeps track of useful metrics

#     In config:
#         - test_average_batch: whether to average responses in batches when computing
#             the test set correlation (used in repeated trials to cancel the neural variability)
#         - compute_oracle_fraction: whether to compute oracle fraction or not
#         - conservative_oracle: whether to compute conservative oracle or not
#         - jackknife_oracle: whether to compute jackknife oracle or not
#         - generate_oracle_figure: whether to generate a figure of fitted line
#             describing the oracle fraction or not
#     """

#     def __init__(self, **config):
#         super().__init__()
#         self.config = config
#         self.test_average_batch = config["test_average_batch"]
#         self.compute_oracle_fraction = config["compute_oracle_fraction"]
#         self.conservative_oracle = config["conservative_oracle"]
#         self.jackknife_oracle = config["jackknife_oracle"]
#         self.generate_oracle_figure = config["generate_oracle_figure"]
#         self.loss = PoissonLoss(avg=True)
#         # self.loss = torch.nn.MSELoss()
#         self.corr = Corr()
#         self.save_hyperparameters()

#     def regularization(self):
#         return 0

#     def training_step(self, batch, batch_idx):
#         """Defines what to do at each training step.
#         Gets the batch, passes it through the network, updates weights,
#         computes loss and regularization, logs important metrics and
#         returns regularized loss.

#         Args:
#             batch (tuple): tuple of (imgs, responses)
#             batch_idx (int): Index of the batch

#         Returns:
#             float: Regularized loss
#         """
#         img, resp = batch
#         prediction = self.forward(img)
#         loss = self.loss(prediction, resp)
#         reg_term = self.regularization()
#         regularized_loss = loss + reg_term
#         self.log("train/unregularized_loss", loss)
#         self.log("train/regularization", reg_term)
#         self.log("train/regularized_loss", regularized_loss)
#         return regularized_loss

#     def validation_step(self, batch, batch_idx):
#         """Defines what to do at each validation step.
#         We just get prediction and return them with target. Later in self.validation_epoch_end,
#         we compute the correlation on the whole validation set (and not on separate
#         batches with final averaging)

#         Args:
#             batch (tuple): tuple of (imgs, responses)
#             batch_idx (int): Index of the batch

#         Returns:
#             tuple: (prediction of responses, true responses)
#         """        

#         img, resp = batch
#         prediction = self.forward(img)
#         loss = self.loss(prediction, resp)
#         self.log("val/loss", loss)

#         return prediction, resp

#     def test_step(self, batch, batch_idx):
#         """
#         - If self.test_average_batch == True, then we average the responses of
#             the batch (because it is the same image shown multiple times to cancel
#             out the noise)

#         - We just get prediction and return them with target. Later in validation_epoch_end,
#             we compute the correlation on the whole validation set (and not on separate
#             batches with final averaging).

#         Args:
#             batch (tuple): tuple of (imgs, responses). The images might be all the
#                 same in case of the oracle dataset for evaluation of the averaged trial correlation.
#             batch_idx (int): Index of the batch

#         Returns:
#             tuple: (prediction of responses, true responses of each trial (might be averaged), true responses of each trial (never averaged))
#         """

#         img, resp = batch
#         responses_no_mean = resp

#         if self.test_average_batch:
#             # I take only one image as all images are the same (it is a repeated trial)
#             # .unsqueeze(0) adds one dimension at the beginning (because I need
#             # to create a batch of size 1)
#             img = img[0].unsqueeze(0)
#             resp = resp.mean(0).unsqueeze(0)

#         prediction = self.forward(img)
#         return prediction, resp, responses_no_mean

#     def configure_optimizers(self):
#         """Configures the optimizer for the training of the model (Adam).

#         Returns:
#             torch.optimizer: torch optimizer class
#         """
#         opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
#         return opt

#     def test_epoch_end(self, test_outs):
#         """We compute a correlation on the WHOLE test set. Predictions with target
#         responses are in test_outs (= what each self.test_step() returned)

#         Args:
#             test_outs (list): What each self.test_step() returned
#         """
#         pred = []
#         resp = []
#         batch_of_responses = []
#         num_of_repeats = None
#         for (p, r, r_batches) in test_outs:

#             # The number of repeats in the repeated trials have to be the same.
#             # We will use the first batch as an indicator, how many trials should
#             # be in every batch. If some batch does not have the same number of
#             # repeats, we discard the batch
#             if num_of_repeats == None:
#                 num_of_repeats = r_batches.shape[0]

#             pred.append(p.detach().cpu().numpy())
#             resp.append(r.detach().cpu().numpy())

#             if (
#                 r_batches.shape[0] == num_of_repeats and self.compute_oracle_fraction
#             ):  # does not have the appropriate number of repeats
#                 batch_of_responses.append(r_batches.detach().cpu().numpy())

#         predictions = np.concatenate(pred)
#         responses = np.concatenate(resp)
#         correlation = corr_from_neuralpredictors(predictions, responses, axis=0)

#         batches_of_responses = None
#         if self.compute_oracle_fraction:
#             batches_of_responses = np.stack(batch_of_responses)

#         if self.test_average_batch:
#             self.log("test/repeated_trials/corr", np.mean(correlation))
#         else:
#             self.log("test/corr", np.mean(correlation))

#         if self.compute_oracle_fraction:
#             if self.jackknife_oracle:
#                 oracles = oracle_corr_jackknife(batches_of_responses)
#                 fraction_of_oracles = get_fraction_oracles(
#                     oracles,
#                     correlation,
#                     generate_figure=self.generate_oracle_figure,
#                     oracle_label="Oracles jackknife",
#                     fig_name="oracle_jackknife.png",
#                 )
#                 self.log("test/fraction_oracle_jackknife", fraction_of_oracles[0])

#             if self.conservative_oracle:
#                 oracles = oracle_corr_conservative(batches_of_responses)
#                 fraction_of_oracles = get_fraction_oracles(
#                     oracles,
#                     correlation,
#                     generate_figure=self.generate_oracle_figure,
#                     oracle_label="Oracles conservative",
#                     fig_name="oracle_conservative.png",
#                 )
#                 self.log("test/fraction_oracle_conservative", fraction_of_oracles[0])

#     def validation_epoch_end(self, val_outs):
#         """We compute the correlation on the whole set. Predictions with target
#         responses are in val_outs (= what each val_step() returned)

#         Args:
#             val_outs (list): What each self.validation_step() returned
#         """        
#         pred = []
#         resp = []
#         for (p, r) in val_outs:
#             pred.append(p.detach().cpu().numpy())
#             resp.append(r.detach().cpu().numpy())

#         predictions = np.concatenate(pred)
#         responses = np.concatenate(resp)
#         correlation = corr_from_neuralpredictors(predictions, responses, axis=0)
#         self.log("val/corr", np.mean(correlation))


# class reCNN_bottleneck_CyclicGauss3d_individual_neuron_scaling(ExtendedEncodingModel):
#     """
#         Like the main model of this repository, but with individual neuron scaling in the readout.
#     """

#     def __init__(self, **config):
#         super().__init__(**config)
#         self.config = config
#         self.nonlinearity = self.config["nonlinearity"]

#         self.hidden_padding = None
#         assert self.config["stack"] == -1

#         self.core = RotationEquivariant2dCoreBottleneck(
#             num_rotations=self.config["num_rotations"],
#             stride=self.config["stride"],
#             upsampling=self.config["upsampling"],
#             rot_eq_batch_norm=self.config["rot_eq_batch_norm"],
#             input_regularizer=self.config["input_regularizer"],
#             input_channels=self.config["input_channels"],
#             hidden_channels=self.config["core_hidden_channels"],
#             input_kern=self.config["core_input_kern"],
#             hidden_kern=self.config["core_hidden_kern"],
#             layers=self.config["core_layers"],
#             gamma_input=config["core_gamma_input"],
#             gamma_hidden=config["core_gamma_hidden"],
#             stack=config["stack"],
#             depth_separable=config["depth_separable"],
#             use_avg_reg=config["use_avg_reg"],
#             bottleneck_kernel=config["bottleneck_kernel"],
#         )

#         self.readout = Gaussian3dCyclic(
#             in_shape=(
#                 self.config["num_rotations"],
#                 self.config["input_size_x"],
#                 self.config["input_size_y"],
#             ),
#             outdims=self.config["num_neurons"],
#             bias=self.config["readout_bias"],
#             mean_activity=self.config["mean_activity"],
#             feature_reg_weight=self.config["readout_gamma"],
#             init_sigma_range=self.config["init_sigma_range"],
#             init_mu_range=self.config["init_mu_range"],
#             fixed_sigma=self.config["fixed_sigma"],
#             do_not_sample=self.config["do_not_sample"],
#         )

#         self.register_buffer("laplace", torch.from_numpy(laplace()))
#         self.nonlin = bl.act_func()[config["nonlinearity"]]

#     def forward(self, x):
#         x = self.core(x)
#         x = self.nonlin(self.readout(x))
#         return x

#     def __str__(self):
#         return "reCNN_bottleneck_CyclicGauss3d_individual_neuron_scaling"

#     def add_bottleneck(self):

#         layer = OrderedDict()

#         if self.hidden_padding is None:
#             self.hidden_padding = self.bottleneck_kernel // 2

#         layer["hermite_conv"] = HermiteConv2D(
#             input_features=self.config["hidden_channels"]
#             * self.config["num_rotations"],
#             output_features=1,
#             num_rotations=self.config["num_rotations"],
#             upsampling=self.config["upsampling"],
#             filter_size=self.config["bottleneck_kernel"],
#             stride=self.config["stride"],
#             padding=self.hidden_padding,
#             first_layer=False,
#         )
#         super().add_bn_layer(layer)
#         super().add_activation(layer)
#         super().features.add_module("bottleneck", nn.Sequential(layer))

#     def regularization(self):

#         readout_l1_reg = self.readout.regularizer(reduction="mean")
#         self.log("reg/readout_l1_reg", readout_l1_reg)

#         readout_reg = readout_l1_reg

#         core_reg = self.core.regularizer()
#         reg_term = readout_reg + core_reg
#         self.log("reg/core reg", core_reg)
#         self.log("reg/readout_reg", readout_reg)
#         return reg_term
    
#     def visualize_orientation_map(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5):
        
#         fig, ax = plt.subplots()
#         x, y, o = get_neuron_estimates(self, 5.5)
#         reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 2.4, 2.4, img_path, suffix, neuron_dot_size)


# class reCNN_bottleneck_CyclicGauss3d_no_scaling(ExtendedEncodingModel):
#     """
#         The main model of this repository.
#         This model consists of:
#             - a core with reCNN architecture with a bottleneck in the last layer
#               to return only one scalar value for each position and orientation
#               (meaning that the number of channels in the last layer is limited
#               to 1)
#             - a readout which is a Gaussian 3d readout but modified in a way
#               that ensures that the third dimension (= orientation dimension)
#               is periodic
#     """

#     def __init__(self, **config):
#         """As this network can be initialized to the ground truth positions and orientations,
#         we need a reference to the dataloader from which this ground truth will be provided.

#         """
#         super().__init__(**config)
#         self.config = config
#         self.nonlinearity = self.config["nonlinearity"]

#         self.hidden_padding = None
#         assert self.config["stack"] == -1

#         self.core = RotationEquivariant2dCoreBottleneck(
#             num_rotations=self.config["num_rotations"],
#             stride=self.config["stride"],
#             upsampling=self.config["upsampling"],
#             rot_eq_batch_norm=self.config["rot_eq_batch_norm"],
#             input_regularizer=self.config["input_regularizer"],
#             input_channels=self.config["input_channels"],
#             hidden_channels=self.config["core_hidden_channels"],
#             input_kern=self.config["core_input_kern"],
#             hidden_kern=self.config["core_hidden_kern"],
#             layers=self.config["core_layers"],
#             gamma_input=config["core_gamma_input"],
#             gamma_hidden=config["core_gamma_hidden"],
#             stack=config["stack"],
#             depth_separable=config["depth_separable"],
#             use_avg_reg=config["use_avg_reg"],
#             bottleneck_kernel=config["bottleneck_kernel"],
#         )

#         self.readout = Gaussian3dCyclicNoScale(
#             in_shape=(
#                 self.config["num_rotations"],
#                 self.config["input_size_x"],
#                 self.config["input_size_y"],
#             ),
#             outdims=self.config["num_neurons"],
#             bias=self.config["readout_bias"],
#             mean_activity=self.config["mean_activity"],
#             feature_reg_weight=self.config["readout_gamma"],
#             init_sigma_range=self.config["init_sigma_range"],
#             init_mu_range=self.config["init_mu_range"],
#             fixed_sigma=self.config["fixed_sigma"],
#             ground_truth_positions_file_path=config["ground_truth_positions_file_path"],
#             ground_truth_orientations_file_path=config["ground_truth_orientations_file_path"],
#             init_to_ground_truth_positions=config["init_to_ground_truth_positions"],
#             init_to_ground_truth_orientations=config["init_to_ground_truth_orientations"],
#             freeze_positions=config["freeze_positions"],
#             freeze_orientations=config["freeze_orientations"],
#             orientation_shift=config["orientation_shift"], #in degrees
#             factor = config["factor"],
#             filtered_neurons = config["filtered_neurons"],
#             positions_minus_x = config["positions_minus_x"],
#             positions_minus_y = config["positions_minus_y"],
#             do_not_sample = config["do_not_sample"],
#         )

#         self.register_buffer("laplace", torch.from_numpy(laplace()))
#         self.nonlin = bl.act_func()[config["nonlinearity"]]
    
#     def init_neurons(self, dataloader=None):
#         self.readout.init_neurons(dataloader)

#     def forward(self, x):
#         x = self.core(x)
#         x = self.nonlin(self.readout(x))
#         return x

#     def __str__(self):
#         return "reCNN_bottleneck_CyclicGauss3d"

#     def add_bottleneck(self):

#         layer = OrderedDict()

#         if self.hidden_padding is None:
#             self.hidden_padding = self.bottleneck_kernel // 2

#         layer["hermite_conv"] = HermiteConv2D(
#             input_features=self.config["hidden_channels"]
#             * self.config["num_rotations"],
#             output_features=1,
#             num_rotations=self.config["num_rotations"],
#             upsampling=self.config["upsampling"],
#             filter_size=self.config["bottleneck_kernel"],
#             stride=self.config["stride"],
#             padding=self.hidden_padding,
#             first_layer=False,
#         )
#         super().add_bn_layer(layer)
#         super().add_activation(layer)
#         super().features.add_module("bottleneck", nn.Sequential(layer))

#     def regularization(self):

#         readout_l1_reg = self.readout.regularizer(reduction="mean")
#         self.log("reg/readout_l1_reg", readout_l1_reg)

#         readout_reg = readout_l1_reg

#         core_reg = self.core.regularizer()
#         reg_term = readout_reg + core_reg
#         self.log("reg/core reg", core_reg)
#         self.log("reg/readout_reg", readout_reg)
#         return reg_term
    
#     def visualize_orientation_map(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5, factor=5.5, shift=0, swap_y_axis=False, neuron_id="all"):
#         """shift is in degrees"""

#         shift = (shift * np.pi) / 180 # from degrees to radians
        
#         fig, ax = plt.subplots()
#         x, y, o = get_neuron_estimates(self, factor)
#         o = [i*np.pi for i in o]
#         o = [(i + shift)%np.pi for i in o]
#         # x, y, o = self.get_ground_truth(ground_truth_positions_file_path, ground_truth_orientations_file_path)

#         if neuron_id != "all":
#             x = x[neuron_id]
#             y = y[neuron_id]
#             o = o[neuron_id]
        
#         print(x)
#         print(y)
#         print(o)
        
#         if swap_y_axis:
#             reconstruct_orientation_maps(x, -y, o, fig, ax, save, 12, 5.5, 5.5, img_path, suffix, neuron_dot_size)
#         else:
#             reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 5.5, 5.5, img_path, suffix, neuron_dot_size)


# # needs_data_loader=True
# class Lurz_Control_Model(ExtendedEncodingModel):
#     """Lurz's model used as a control model"""

#     def __init__(self, **config):
#         super().__init__(**config)
#         self.config = config
#         self.nonlinearity = self.config["nonlinearity"]
        

#         self.core = cores.SE2dCore(
#             stride=self.config["stride"],
#             input_regularizer=self.config["input_regularizer"],
#             input_channels=self.config["input_channels"],
#             hidden_channels=self.config["core_hidden_channels"],
#             input_kern=self.config["core_input_kern"],
#             hidden_kern=self.config["core_hidden_kern"],
#             layers=self.config["core_layers"],
#             gamma_input=config["core_gamma_input"],
#             gamma_hidden=config["core_gamma_hidden"],
#             stack=config["stack"],
#             depth_separable=config["depth_separable"],
#             use_avg_reg=config["use_avg_reg"]
#         )

        
#         self.readout = readouts.FullGaussian2d(
#             in_shape=(
#                 self.config["core_hidden_channels"] * abs(self.config["stack"]),
#                 self.config["input_size_x"],
#                 self.config["input_size_y"],
#             ),
#             outdims=self.config["num_neurons"],
#             bias=self.config["readout_bias"],
#             mean_activity=self.config["mean_activity"],
#             feature_reg_weight=self.config["readout_gamma"],
#             init_sigma=self.config["init_sigma_range"],
#         )

#         self.init_to_ground_truth_positions = config["init_to_ground_truth_positions"]
#         self.ground_truth_positions_file_path = config["ground_truth_positions_file_path"]
#         self.ground_truth_orientations_file_path = config["ground_truth_orientations_file_path"]
#         self.positions_minus_x = config["positions_minus_x"]
#         self.positions_minus_y = config["positions_minus_y"]
#         self.do_not_sample = config["do_not_sample"]
#         self.positions_swap_axes = config["positions_swap_axes"]
        
#         self.register_buffer("laplace", torch.from_numpy(laplace()))
#         self.nonlin = bl.act_func()[config["nonlinearity"]]


#     def init_neurons(self, dataloader=None):

#         if self.init_to_ground_truth_positions == True:
#             print("initializing to ground truth")
#             pos_x, pos_y, _ = dataloader.get_ground_truth(ground_truth_positions_file_path=self.ground_truth_positions_file_path, ground_truth_orientations_file_path=self.ground_truth_orientations_file_path, in_degrees=True, positions_minus_y=self.positions_minus_y, positions_minus_x=self.positions_minus_x, positions_swap_axes=self.positions_swap_axes)
#             pos_x = torch.from_numpy(pos_x)
#             pos_y = torch.from_numpy(pos_y)
#             # works also when the stimulus is cropped (self.get_stimulus_visual_angle()
#             # returns the visual angle corrected after the stimulus crop)
#             self.readout._mu.data[0,:,0,0] = pos_x / (dataloader.get_stimulus_visual_angle() / 2)        
#             self.readout._mu.data[0,:,0,1] = pos_y / (dataloader.get_stimulus_visual_angle() / 2)

#     def forward(self, x):
#         x = self.core(x)
#         x = self.nonlin(self.readout(x))
#         return x
    
#     def __str__(self):
#         return "StackedCore_FullGaussian2d"

#     def reg_readout_group_sparsity(self):
#         nw = self.readout.features.reshape(self.config["num_neurons"], -1)
#         reg_term = self.config["reg_group_sparsity"] * torch.sum(
#             torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
#         )
#         return reg_term

#     def regularization(self):

#         readout_l1_reg = self.readout.regularizer(reduction="mean")
#         self.log("reg/readout_l1_reg", readout_l1_reg)

#         readout_reg = readout_l1_reg
#         core_reg = self.core.regularizer()
#         reg_term = readout_reg + core_reg
#         self.log("reg/core reg", core_reg)
#         self.log("reg/readout_reg", readout_reg)
#         return reg_term


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
        pnl_vmin = 0, 
        pnl_vmax = 30, 
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
        self.f = torch.nn.Parameter(torch.rand(1) + f_init)


        self.filter_scale = torch.nn.Parameter(torch.ones(1) * filter_scale_init)
        self.final_scale = torch.nn.Parameter(torch.ones(1) * final_scale_init)

        if common_rescaling==True:
            self.final_bias = torch.nn.Parameter(torch.ones(1)*5)
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
        self.nonlinearity = nonlinearity
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
        x_shape = x.shape
        
        x = self.nonlinearity(x.reshape(-1, 1)).reshape(x_shape)
        x = self.final_scale * x + self.final_bias
        return x

    def regularizer(self, data_key=None, **kwargs):
        return 0

# %%




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
