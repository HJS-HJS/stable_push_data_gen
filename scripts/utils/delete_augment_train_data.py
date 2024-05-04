import os
import re
import parmap
import multiprocessing
import numpy as np

# Configure paths
PATH = os.getcwd()
data_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data/tensors"

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = int(indices[-1]/2)

def delete_data(idx):
    name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(max_index + idx + 1)
    os.remove(data_dir + '/image' + name)
    os.remove(data_dir + '/masked_image' + name)
    os.remove(data_dir + '/velocity' + name)
    os.remove(data_dir + '/label' + name)

num_cores = multiprocessing.cpu_count()
parmap.map(delete_data, range(max_index), pm_pbar={'desc': 'Saving flipped data'}, pm_processes=num_cores, pm_chunksize=num_cores)
