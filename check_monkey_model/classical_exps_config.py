#%%

import numpy as np
import sys
from experiments import * 
from monkey_training.load_best_model import get_best_brcnn_model
from classical_exps.functions.utils import plot_img, pickleread
from classical_exps.functions.experiments import *

main_dir = "/project/"
run_dir = '/project/check_monkey_model'
objects_dir = main_dir + "/objects"

## Name of the HDF5 file 
h5_file = run_dir + "/results_convnext_model.h5"

## Select the model with all neurons
# fix this
all_neurons_model = get_best_brcnn_model(subject='both')
all_neurons_model.eval()

## Chose the indices of the neurons to work with
# fix this
n = 458
neuron_ids = np.arange(0, n, 40)
# corrs = pickleread(objects_dir + "/avg_corr.pkl") ## The correlation score of the neurons
# neuron_ids = neuron_ids[corrs>0.75]

## Grating parameters to test
orientations = np.linspace(0, np.pi, 37)[:-1] 
spatial_frequencies = np.linspace(0.5, 1.5, 5) 
phases = np.linspace(0, 2*np.pi, 37)[:-1]

## Parameters for the dot stimulation
dot_size_in_pixels_gauss = 2
bs = 40
seed = 0

## Fixed image parameters
contrast = 1         
img_res = [46,46] 
pixel_min = -1.7876 # (the lowest value encountered in the data that served to train the model, serves as the black reference)
pixel_max =  2.1919 # (the highest [...] serves at white reference )
size = 2.67 

## Allow negative responses (because the response we show is the difference between the actual response and the response to a gray screen)
neg_val = True ## Keep True

## EXPERIMENTS 1 arguments
radii = np.logspace(-2,np.log10(2),40)
## For the contrast response experiment
center_contrasts = np.logspace(np.log10(0.06),np.log10(1),18) 
surround_contrasts = np.logspace(np.log10(0.06),np.log10(1),6)

## EXPERIMENTS 2 arguments
## For the orientation tuning experiment
ori_shifts = np.linspace(-np.pi,np.pi,9)
## For the ccss experiment
center_contrasts_ccss = np.array([0.06,0.12,0.25,0.5,1.0,1.5])
surround_contrast = contrast

## EXPERIMENTS 3 arguments
contrasts = np.logspace(np.log10(0.06),np.log10(1),5)  ## Same as in the article
dot_size_in_pixels = 5  

## EXPERIMENTS 4 arguments
directory_imgs = objects_dir + '/shareStim_NN13'
target_res = img_res
num_samples = 15

## If overwrite is set to True, this will clean the results of the performed experiments before reperforming them
overwrite = False 

## (optional) Device to perform the experiments on (default will be gpu if available, cpu else)
device=None

## Example Pipeline
experiments_config = [
    # ['get_all_grating_parameters', {
    #     'h5_file':h5_file, 
    #     'all_neurons_model':all_neurons_model, 
    #     'neuron_ids':neuron_ids, 
    #     'overwrite':True, 
    #     'orientations':orientations, 
    #     'spatial_frequencies':spatial_frequencies, 
    #     'phases':phases, 'contrast':contrast,
    #     'img_res':img_res,
    #     'pixel_min':pixel_min,
    #     'pixel_max':pixel_max,
    #     'device':device,
    #     'size':size}], 
    # ['get_preferred_position', {
    #     'h5_file':h5_file,
    #     'all_neurons_model':all_neurons_model,
    #     'neuron_ids':neuron_ids,
    #     'overwrite':True,
    #     'dot_size_in_pixels':dot_size_in_pixels_gauss,
    #     'contrast':contrast,
    #     'img_res':img_res,
    #     'pixel_min':pixel_min,
    #     'pixel_max':pixel_max,
    #     'device':device,
    #     'bs':bs,
    #     'seed':seed}], 
    ['size_tuning_experiment_all_phases', {
        'h5_file':h5_file,
        'all_neurons_model':all_neurons_model,
        'neuron_ids':neuron_ids,
        'overwrite':True,
        'radii':radii,
        'phases':phases,
        'contrast':contrast,
        'pixel_min':pixel_min,
        'pixel_max':pixel_max,
        'device':device,
        'size':size,
        'img_res':img_res,
        'neg_val':neg_val}],
    # ['contrast_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts':center_contrasts, 'surround_contrasts':surround_contrasts, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['contrast_size_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'contrasts':contrasts, 'radii':radii, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res, 'neg_val':neg_val}],
    # ['orientation_tuning_experiment_all_phases', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'phases':phases, 'ori_shifts':ori_shifts, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['center_contrast_surround_suppression_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'center_contrasts_ccss':center_contrasts_ccss, 'surround_contrast':surround_contrast, 'phases':phases, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'size':size, 'img_res':img_res}],
    # ['black_white_preference_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'overwrite':overwrite, 'dot_size_in_pixels':dot_size_in_pixels, 'contrast':contrast, 'img_res':img_res, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'device':device, 'seed':seed}],
    # ['texture_noise_response_experiment', {'h5_file':h5_file, 'all_neurons_model':all_neurons_model, 'neuron_ids':neuron_ids, 'directory_imgs':directory_imgs, 'overwrite':overwrite, 'contrast':contrast, 'pixel_min':pixel_min, 'pixel_max':pixel_max, 'num_samples':num_samples, 'img_res':img_res, 'device':device}]
    ]


## Filtering parameters
fit_err_thresh = 0.2
supp_thresh = 0.1
SNR_thresh = 2

##size_tuning_results_2

sort_by_std = False
spread_to_plot = [0,15,50,85,100]

## contrast_size_tuning_results_1
shift_to_plot = spread_to_plot
low_contrast_id  =  0
high_contrast_id = -1

## Center Contrast Surround Suppression
high_center_contrast_id      = -3 ## The contrast of the center corresponding to the 'high' contrast
high_norm_center_contrast_id = high_center_contrast_id
low_center_contrast_id       = 1  ## The contrast of the center corresponding to the 'low' contrast
low_norm_center_contrast_id  = low_center_contrast_id  

## black_white_results_1
# neuron_depths = pickleread(objects_dir + "/depth_info.pickle") 

## texture_noise_response_results
wanted_fam_order = ['60', '56', '13', '48', '71', '18', '327', '336', '402', '38', '23', '52', '99', '393', '30'] ## The order of the textures

## Example Pipeline
neuron_id = 0

analyses_config = [
    # ['plot_size_tuning_curve', {'h5_file':h5_file, 'neuron_id':neuron_id}],
    # ['size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'supp_thresh':supp_thresh}],
    # ['size_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['contrast_response_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'sort_by_std':sort_by_std, 'spread_to_plot':spread_to_plot}],
    # ['contrast_size_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh, 'shift_to_plot':shift_to_plot, 'low_contrast_id':low_contrast_id, 'high_contrast_id':high_contrast_id}],
    # ['orientation_tuning_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['orientation_tuning_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':high_center_contrast_id, 'norm_center_contrast_id':high_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['ccss_results_2', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'contrast_id':low_center_contrast_id, 'norm_center_contrast_id':low_norm_center_contrast_id, 'fit_err_thresh':fit_err_thresh}],
    # ['black_white_results_1', {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'neuron_depths':neuron_depths, 'SNR_thresh':SNR_thresh}],
    # ['texture_noise_response_results_1',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_2',  {'h5_file':h5_file, 'neuron_ids':neuron_ids, 'wanted_fam_order':wanted_fam_order}],
    # ['texture_noise_response_results_3',  {'h5_file':h5_file, 'neuron_ids':neuron_ids}]
]


def execute_function(func_name, params):
    # Get the function object by name
    func = globals().get(func_name)
    
    # Check if the function exists
    if func is None or not callable(func):
        raise ValueError(f"Function '{func_name}' not found or not callable.")
    
    # Call the function with the parameters
    func(**params)
# %%
