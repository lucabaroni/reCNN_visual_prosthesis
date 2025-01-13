#%%
import numpy as np
import os
from tqdm import tqdm
directory = '/CSNG/baroni/Dic23data/single_trial'

x =sorted(os.listdir(directory))
# %%
stims = np.zeros([len(x), *np.load(directory + '/' + x[0]  + '/stimulus.npy').shape])
for i in tqdm(np.arange(len(x))):
    stims[i] = np.load(directory + '/' + x[i]  + '/stimulus.npy')
np.save('/CSNG/baroni/Dic23data/all_single_trial_stims.npy', stims)


# resp_exc = np.zeros([len(x), *np.load(directory + '/' + x[0]  + '/V1_Exc_L23.npy').shape])
# for i in tqdm(np.arange(len(x))):
#     resp_exc[i] = np.load(directory + '/' + x[i]  + '/V1_Exc_L23.npy')
# np.save('/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy', resp_exc)


# resp_inh = np.zeros([len(x), *np.load(directory + '/' + x[0]  + '/V1_Inh_L23.npy').shape])
# for i in tqdm(np.arange(len(x))):
#     resp_inh[i] = np.load(directory + '/' + x[i]  + '/V1_Inh_L23.npy')
# np.save('/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy', resp_inh )

