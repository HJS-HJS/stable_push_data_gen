import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
from utils.push_dof_tools import get_maximum_file_idx
from utils.dataloader_parallel import DataLoaderParallel
import parmap
import multiprocessing

import re

# Parse arguments
parser = argparse.ArgumentParser(description='This script augment Isaac Gym asset.')
parser.add_argument('--var', required=True, help='var')
parser.add_argument('--config_file', required=True, help='Path to urdf folder')
parser.add_argument('--train_data_dir', required=True, help='Path to urdf folder')
args = parser.parse_args()

config_dir = args.config_file
var = args.var
train_data_dir = args.train_data_dir

# Open config file
with open(config_dir,'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

FILE_NUM_ZERO_PADDING = cfg["simulation"]["FILE_ZERO_PADDING_NUM"]

file_list = os.listdir(train_data_dir)
file_list = [file for file in file_list if file.startswith(var)]
indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = indices[-1]
print(var, ":", max_index)

def save_data(idx):
    old_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(idx)
    new_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(max_index + idx + 1)
        
    data = np.load(os.path.join(train_data_dir, var + old_name), allow_pickle=True)

    if (var == "image") or (var == "masked_image"):
        with open(os.path.join(train_data_dir, var + new_name), 'wb') as f:
            np.save(f, np.flip(data, axis=1))
    elif var == "velocity":
        with open(os.path.join(train_data_dir, var + new_name), 'wb') as f:
            np.save(f, np.array([-data[0], data[1], data[2]]))
    else:
        with open(os.path.join(train_data_dir, var + new_name), 'wb') as f:
            np.save(f, data)



num_cores = multiprocessing.cpu_count()
parmap.map(save_data, range(max_index), pm_pbar={'desc': 'Move ' + var + ' data'}, pm_processes=num_cores, pm_chunksize=num_cores)


# # Load training data
# image_list = dataloader.load_image_tensor_parallel()
# masked_image_list = dataloader.load_masked_image_tensor_parallel()
# velocity_list = dataloader.load_velocity_tensor_parallel()
# label_list = dataloader.load_label_tensor_parallel()
# # origin_image_list = dataloader.load_tensor_parallel("masked_origin_image")


# # Get file list by each field
# file_list_image = [file for file in file_list if file.startswith('image')]
# file_list_masked_image = [file for file in file_list if file.startswith('masked_image')]
# file_list_velocity = [file for file in file_list if file.startswith('velocity')]
# file_list_labels = [file for file in file_list if file.startswith('label')]
# # file_list_origin_image = [file for file in file_list if file.startswith('masked_origin_image')]
    
# images = np.squeeze(np.array(image_list), axis=1)
# masked_images = np.squeeze(np.array(masked_image_list), axis=1)
# velocities = np.array(velocity_list)
# labels = np.array(label_list)
# # origin_images = np.array(origin_image_list)

# # # Flip train data
# flipped_images = np.flip(images, axis=2)
# flipped_masked_images = np.flip(masked_images, axis=2)
# # flipped_origin_images = np.flip(origin_images, axis=2)

# # print(data_max_idx)

# def save_data(idx):
#     name = ("_%0" + str(num_zero_padding) + 'd.npy')%(data_max_idx + idx + 1)
    
#     with open(os.path.join(train_data_dir, 'image' + name), 'wb') as f:
#         np.save(f, flipped_images[idx])
        
#     with open(os.path.join(train_data_dir, 'masked_image' + name), 'wb') as f:
#         np.save(f, flipped_masked_images[idx])
        
#     with open(os.path.join(train_data_dir, 'velocity' + name), 'wb') as f:
#         np.save(f, np.array([-velocities[idx][0], velocities[idx][1], velocities[idx][2]]))
        
#     with open(os.path.join(train_data_dir, 'label' + name), 'wb') as f:
#         np.save(f, labels[idx])

#     # with open(os.path.join(train_data_dir, 'masked_origin_image' + name), 'wb') as f:
#         # np.save(f, flipped_origin_images[idx])

# num_cores = multiprocessing.cpu_count()
# parmap.map(save_data, range(len(flipped_masked_images)), pm_pbar={'desc': 'Saving flipped data'}, pm_processes=num_cores, pm_chunksize=num_cores)
# print('convert ', len(flipped_masked_images), 'data')