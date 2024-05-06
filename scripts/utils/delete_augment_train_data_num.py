import os
import re
import parmap
import multiprocessing
import numpy as np

DELETE_IDX = 168469

# Configure paths
PATH = os.getcwd()
data_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data/tensors"

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = int(indices[-1])

print('total data {}'.format(max_index))

def delete_data(idx):
    if idx < (max_index - DELETE_IDX):
        return
    name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(idx + 1)
    os.remove(data_dir + '/image' + name)
    os.remove(data_dir + '/masked_image' + name)
    os.remove(data_dir + '/masked_origin_image' + name)
    os.remove(data_dir + '/velocity' + name)
    os.remove(data_dir + '/label' + name)

num_cores = multiprocessing.cpu_count()
parmap.map(delete_data, range(max_index), pm_pbar={'desc': 'Saving flipped data'}, pm_processes=num_cores, pm_chunksize=num_cores)

print('delete {} from {} data {}'.format(DELETE_IDX, max_index, max_index - DELETE_IDX))