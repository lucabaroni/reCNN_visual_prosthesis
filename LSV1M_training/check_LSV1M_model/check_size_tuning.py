#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import pickleread
import torch
from classical_exps_config import radii
import matplotlib.pyplot as plt

n = 25
# Path to your HDF5 file
file_path = '/home/baroni/recnn/LSV1M_training/check_LSV1M_model/new_results_LSV1M_model_new_model.h5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as h5file:
    # Check if the 'size_tuning/curves' group exists
    if 'size_tuning/curves' in h5file:
        group = h5file['size_tuning/curves']
        
        # Iterate over neurons from 0 to 99
        for neuron_id in range(n):
            dataset_name = f'neuron_{neuron_id}'
            
            if dataset_name in group:
                dataset = group[dataset_name]
                
                # Extract the first row
                data_to_plot = dataset[0, :]
                
                # Plot the data
                plt.plot(radii, data_to_plot, label=f'Neuron {neuron_id}')
                      # Add labels and title to the plot
                plt.xlabel('radius')
                plt.ylabel('activation')
                plt.title(f'neuron_id: {neuron_id}')
                plt.legend(loc='best')
                plt.show()
                plt.show()
            else:
                print(f"Dataset '{dataset_name}' not found in 'size_tuning/curves'.")
        
    else:
        print("'size_tuning/curves' group not found in the file.")
d# %%
