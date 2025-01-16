#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import pickleread
import torch
pos_ori = pickleread('/CSNG/baroni/Dic23data/pos_and_ori.pkl')
pos1 = np.concatenate([pos_ori['V1_Exc_L23']['pos_x'], pos_ori['V1_Inh_L23']['pos_x']]) / (5.5)
pos2 = np.concatenate([pos_ori['V1_Exc_L23']['pos_y'], pos_ori['V1_Inh_L23']['pos_y']]) / (5.5)
ori = np.concatenate([pos_ori['V1_Exc_L23']['ori'], pos_ori['V1_Inh_L23']['ori']]) / np.pi

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


####### INSPECT POSITIONS #################
# Open the HDF5 file
file_path = '/project/check_trained_model/results_LSV1M_model.h5'
with h5py.File(file_path, 'r') as h5file:
    
    # Initialize a list to hold the datasets
    datasets = []
    
    # Iterate through the dataset names
    for i in range(100):  # Assuming 'neuron_x' goes from 0 to 200
        dataset_name = f'preferred_pos/neuron_{i}'
        if dataset_name in h5file:
            # Load the dataset into a NumPy array and append to the list
            datasets.append(h5file[dataset_name][:])
        else:
            print(f"Dataset {dataset_name} not found in the file.")

    # Stack all the datasets along a new axis (e.g., axis 0)
    stacked_data = np.stack(datasets, axis=0)

# Optionally, print the shape of the stacked array to confirm
print(f"Stacked data shape: {stacked_data.shape}")

idxx = (stacked_data[:,0]!=27.5) * (stacked_data[:,2] < 1)
x = stacked_data[:,0]
idxy = (stacked_data[:,1]!=27.5) * (stacked_data[:,2] < 1)
y = stacked_data[:,1]

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
# Ensure the axes are square
plt.gca().set_aspect(2/55, adjustable='box')
plt.show()


# %% ORI 
with h5py.File(file_path, 'r') as h5file:
    
    # Initialize a list to hold the datasets
    datasets = []
    
    # Iterate through the dataset names
    for i in range(100):  # Assuming 'neuron_x' goes from 0 to 200
        dataset_name = f'full_field_params/neuron_{i}'
        if dataset_name in h5file:
            # Load the dataset into a NumPy array and append to the list
            datasets.append(h5file[dataset_name][:])
        else:
            print(f"Dataset {dataset_name} not found in the file.")

    # Stack all the datasets along a new axis (e.g., axis 0)
    stacked_data = np.stack(datasets, axis=0)

# Optionally, print the shape of the stacked array to confirm
print(f"Stacked data shape: {stacked_data.shape}")
# %%
ori2 = stacked_data[:,0]
# %%

plt.scatter(ori[:100], ori2 )
plt.xlim(0, 1)
plt.ylim(0, np.pi)
plt.title('Orientation')
plt.xlabel('Orientation in model')
plt.ylabel('Orientation of maximally exciting full field grating')
plt.gca().set_aspect(1/np.pi, adjustable='box')
plt.show()

#%%

# %%
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

full_field_params = get_stacked_datasets(file_path, 'full_field_params')

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


# %%
