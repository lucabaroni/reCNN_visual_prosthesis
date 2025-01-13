#%%
import numpy as np
import os
from tqdm import tqdm

# up_to_50k_stims = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
# from50k_stims = np.load('/CSNG/baroni/Dic23data/all_single_trial_stims.npy')
# stims100k = np.concatenate([up_to_50k_stims, from50k_stims])
# np.save(stims100k, '/CSNG/baroni/Dic23data/100k_single_trial_stims.npy')
# %%
up_to_50k_Exc = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
from50k_Exc = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy')
Exc100k = np.concatenate([up_to_50k_Exc, from50k_Exc])
np.save(Exc100k, '/CSNG/baroni/Dic23data/100k_single_trial_V1_Exc_L23.npy')
# #%%
# up_to_50k_Inh = np.stack([np.load(f'/CSNG/baroni/Dic23data/single_trial/{idx:010d}/stimulus.npy') for idx in tqdm(np.arange(0, 50000))])
# from50k_Inh = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy')
# Inh100k = np.concatenate([up_to_50k_Inh, from50k_Inh])
# np.save(Inh100k, '/CSNG/baroni/Dic23data/100k_single_trial_V1_Inh_L23.npy')