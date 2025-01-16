#%%
import numpy as np
#%%
# single trial
print('Preprocessing single trial data...')

print('Loading responses...')
resps_exc = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23.npy')
print('Normalizing responses...')
resps_exc_mean = resps_exc.mean(axis=0, keepdims=True)
print('done')
resps_exc_std = resps_exc.std(axis=0, keepdims=True)
print('done')
resps_exc = (resps_exc - resps_exc_mean)/resps_exc_std
print('done')
np.save('/CSNG/baroni/Dic23data/all_single_trial_V1_Exc_L23_preprocessed.npy', resps_exc)
del resps_exc
#%%
print('Loading responses...')
resps_inh = np.load('/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23.npy')
print('Normalizing responses...')
resps_inh_mean = resps_inh.mean(axis=0, keepdims=True)
resps_inh_std = resps_inh.std(axis=0, keepdims=True)
resps_inh = (resps_inh - resps_inh_mean)/resps_inh_std
np.save('/CSNG/baroni/Dic23data/all_single_trial_V1_Inh_L23_preprocessed.npy', resps_inh)
del resps_inh

#%%
print('Loading stimuli...')
stims = np.load('/CSNG/baroni/Dic23data/all_single_trial_stims55x55.npy')
print('Normalizing stimuli...')
stims_mean = stims.mean()
stims_std = stims.std()     
stims = (stims - stims_mean)/stims_std
np.save('/CSNG/baroni/Dic23data/all_single_trial_stims55x55_preprocessed_non_centered.npy', stims)
print('Centering stimuli...')
stims_max = stims.max()
stims_min = stims.min()
stims = stims - (stims_max + stims_min)/2
np.save('/CSNG/baroni/Dic23data/all_single_trial_stims55x55_preprocessed_centered.npy', stims)
del stims

#%%
# multi trial
print('Preprocessing multi trial data...')
print('Loading responses...')
resps_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_resps.npy')
print('Normalizing responses...')
resps_multi_mean = np.concatenate([resps_exc_mean, resps_inh_mean], axis=-1).reshape(1,1,-1)
resps_multi_std = np.concatenate([resps_exc_std, resps_inh_std], axis=-1).reshape(1,1,-1)
resps_multi = (resps_multi - resps_multi_mean)/resps_multi_std
np.save('/CSNG/baroni/Dic23data/all_multi_trial_resps_preprocessed.npy', resps_multi)
del resps_multi

print('Loading stimuli...')
stims_multi = np.load('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55.npy')
print('Normalizing stimuli...')
stims_multi = (stims_multi - stims_mean)/stims_std
np.save('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55_preprocessed_non_centered.npy', stims_multi)
print('Centering stimuli...')
stims_multi = stims_multi - (stims_multi.max() + stims_multi.min())/2
np.save('/CSNG/baroni/Dic23data/all_multi_trial_stims55x55_preprocessed_centered.npy', stims_multi)
del stims_multi

# %%
