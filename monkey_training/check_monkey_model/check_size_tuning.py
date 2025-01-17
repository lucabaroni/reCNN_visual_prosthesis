#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pickle_utils import pickleread
import torch
from classical_exps_config import radii
import matplotlib.pyplot as plt

# Path to your HDF5 file
file_path = '/project/check_monkey_model/results_convnext_model.h5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as h5file:
    # Check if the 'size_tuning/curves' group exists
    if 'size_tuning/curves' in h5file:
        group = h5file['size_tuning/curves']
        
        # Iterate over neurons from 0 to 99
        for neuron_id in range(0,458, 40):
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

        # %%

#%%
# Path to your HDF5 file
file_path = '/project/check_monkey_model/results_convnext_model.h5'

# Open the HDF5 file in read mode
with h5py.File(file_path, 'r') as h5file:
    # Check if the 'size_tuning/curves' group exists
    if 'size_tuning/curves' in h5file:
        group = h5file['size_tuning/curves']
        
        # Store data for all neurons
        all_data = []
        
        # Iterate over neurons
        for neuron_id in range(0,458):
            dataset_name = f'neuron_{neuron_id}'
            
            if dataset_name in group:
                dataset = group[dataset_name]
                # Extract the first row
                data = dataset[0, :]
                all_data.append(data)
            else:
                print(f"Dataset '{dataset_name}' not found in 'size_tuning/curves'.")
        
        # Convert to numpy array
        all_data = np.array(all_data)
        
        # Calculate mean and standard error
        mean_data = np.mean(all_data, axis=0)
        sem_data = np.std(all_data, axis=0) / np.sqrt(len(all_data))
        
        # Find max value and index
        max_idx = np.argmax(mean_data)
        max_val = mean_data[max_idx]
        
        # Find minimum value after maximum
        min_after_max = np.min(mean_data[max_idx:])
        
        # Plot mean with shaded error using seaborn
        import seaborn as sns
        plt.figure(figsize=(10,6))
        sns.lineplot(x=radii, y=mean_data, color='#2E86AB')  # Rich blue for main line
        plt.fill_between(radii, mean_data-sem_data, mean_data+sem_data, alpha=0.3, color='#A8D0DB')  # Light blue for shading
        plt.hlines(y=max_val, xmin=0, xmax=1.3, color='#24305E', linestyle='--', label='Max Activation')  # Dark navy
        plt.hlines(y=min_after_max, xmin=0, xmax=1.3, color='#F76C6C', linestyle='--', label='Min After Max')  # Coral red
        plt.xlabel('Radius')
        plt.ylabel('Activation')
        plt.title('Mean Size Tuning Curve with SEM')
        plt.xlim(0, 1.3)
        plt.legend()
        plt.show()
    else:
        print("'size_tuning/curves' group not found in the file.")
 # %%
