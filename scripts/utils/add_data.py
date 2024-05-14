import os
import re
import shutil
import parmap
import multiprocessing
import numpy as np

# Configure paths
PATH = os.getcwd()
data_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data/tensors"
data_add_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data_add/tensors"

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = indices[-1]

try:
    file_list = os.listdir(data_add_dir)
except:
    raise Exception("Directory Not Exist: ", data_add_dir)
file_list = [file for file in file_list if file.endswith('.npy')]
indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
try:
    max_add_index = indices[-1]
    min_add_index = indices[0]
except:
    raise Exception("Data Not Exists")

def move_data(idx):
    old_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(min_add_index + idx)
    new_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(max_index + idx + 1)
    shutil.move(os.path.join(data_add_dir, 'image' + old_name), os.path.join(data_dir, 'image' + new_name))
    shutil.move(os.path.join(data_add_dir, 'masked_image' + old_name), os.path.join(data_dir, 'masked_image' + new_name))
    shutil.move(os.path.join(data_add_dir, 'velocity' + old_name), os.path.join(data_dir, 'velocity' + new_name))
    shutil.move(os.path.join(data_add_dir, 'label' + old_name), os.path.join(data_dir, 'label' + new_name))

num_cores = multiprocessing.cpu_count()
parmap.map(move_data, range(max_add_index + 1 - min_add_index), pm_pbar={'desc': 'Move data'}, pm_processes=num_cores, pm_chunksize=num_cores)
