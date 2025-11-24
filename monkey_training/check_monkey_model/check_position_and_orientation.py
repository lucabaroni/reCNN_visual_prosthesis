#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import pickleread
import torch

h5 = '/project/check_monkey_model/results_convnext_model.h5'
data1 = {i: {} for i in range(458)}
with h5py.File(h5, 'r') as f:
    for i in range(458):
        dataset_name = f'full_field_params/neuron_{i}'
        data1[i]['preferred_ori'] = f[dataset_name][:][0]
        dataset_name = f'preferred_pos/neuron_{i}'
        data1[i]['preferred_pos'] = f[dataset_name][:]
pos1 = []
pos2 = []
ori = []
for i in range(458):
    pos1.append((data1[i]['preferred_pos'][0]-(93/2))/(93/2))
    pos2.append((data1[i]['preferred_pos'][1]-(93/2))/(93/2))
    ori.append(data1[i]['preferred_ori']/np.pi)
pos1 = torch.Tensor(pos1)
pos2 = torch.Tensor(pos2)
ori = torch.Tensor(ori)

def get_stacked_datasets(file_path, group, idxs=np.arange(100)):
    with h5py.File(file_path, 'r') as h5file:
    
        # Initialize a list to hold the datasets
        datasets = []
        # Iterate through the dataset names
        for i in idxs:  # Assuming 'neuron_x' goes from 0 to 200
            dataset_name = f'{group}/neuron_{i}'
            if dataset_name in h5file:
                # Load the dataset into a NumPy array and append to the list
                datasets.append(h5file[dataset_name][:])
            else:
                print(f"Dataset {dataset_name} not found in the file.")

        # Stack all the datasets along a new axis (e.g., axis 0)
        stacked_data = np.stack(datasets, axis=0)
    return stacked_data

file_path = '/project/check_monkey_model/results_convnext_model.h5'
full_field_params = get_stacked_datasets(file_path, 'full_field_params')


def get_stacked_datasets_grating(file_path, group, idxs=np.arange(100)):
    with h5py.File(file_path, 'r') as h5file:
    
        # Initialize a list to hold the datasets
        datasets = []
        # Iterate through the dataset names
        for i in idxs:  # Assuming 'neuron_x' goes from 0 to 200
            dataset_name = f'{group}/neuron_{i}_grating'
            if dataset_name in h5file:
                # Load the dataset into a NumPy array and append to the list
                datasets.append(h5file[dataset_name][:])
            else:
                print(f"Dataset {dataset_name} not found in the file.")

        # Stack all the datasets along a new axis (e.g., axis 0)
        stacked_data = np.stack(datasets, axis=0)
    return stacked_data

grating = get_stacked_datasets_grating(file_path, 'full_field_params')

# %%
ori2 = full_field_params[:,0]
frequency = full_field_params[:,1]
phase = full_field_params[:,2]

#%%
plt.hist(frequency)
plt.show()

# %%
plt.scatter(ori[:100], ori2, c=frequency, cmap='viridis')
plt.colorbar()
plt.show()
# %% ########## NEW ##################

##%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import pickleread
import torch

file_path = '/project/check_trained_model/results_LSV1M_model_new.h5'

pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']]) / (5.5)
pos2 = np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']]) / (5.5)
ori = np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']]) / np.pi

positions = get_stacked_datasets(file_path, 'preferred_pos')
idxx = (positions[:,0]!=27.5) * (positions[:,2] < 1)
x = positions[:,0]
idxy = (positions[:,1]!=27.5) * (positions[:,2] < 1)
y = positions[:,1]
error = positions[:,2]

plt.scatter(pos1[:100][idxx], x[idxx])
plt.ylim(0, 55)
plt.xlim(-1,1)
plt.title('x axis')
plt.xlabel('Position in model')
plt.ylabel('Gaussian fit after dot stimulation')
plt.gca().set_aspect(2/55, adjustable='box')
plt.show()

plt.scatter(pos2[:100][idxy], y[idxy])
plt.ylim(0, 55)
plt.xlim(-1,1)
plt.title('y axis')
plt.xlabel('Position in model')
plt.ylabel('Gaussian fit after dot stimulation')
plt.gca().set_aspect(2/55, adjustable='box')
plt.show()

plt.hist(error)
plt.show()
# %% ORI 
full_field_grating_params = get_stacked_datasets(file_path, 'full_field_params')
ori2 = full_field_grating_params[:,0]

plt.scatter(ori[:100], ori2 )
plt.xlim(0, 1)
plt.ylim(0, np.pi)
plt.title('Orientation')
plt.xlabel('Orientation in model')
plt.ylabel('Orientation of maximally exciting full field grating')
plt.gca().set_aspect(1/np.pi, adjustable='box')
plt.show()

ori2 = full_field_grating_params[:,0]
frequency = full_field_grating_params[:,1]
phase = full_field_grating_params[:,2]

plt.scatter(ori[:100], ori2, c=frequency, cmap='viridis')
plt.colorbar()
plt.show()
plt.hist(frequency)
plt.title('frequency')
plt.show()
plt.hist(phase)
plt.title('phase')
plt.show()
plt.hist(ori)
plt.title('ori')
plt.show()
# %%
preferred_pos = get_stacked_datasets(file_path, 'preferred_pos')
# ori2 = full_field_grating_params[:,0]

# %%
preferred_pos
# %%
