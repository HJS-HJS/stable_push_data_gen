import os
import re
import argparse
import numpy as np
from utils.dataloader_parallel import DataLoaderParallel

'''
Derives mean and standard deviation values from train data.

- Because of large volume of training data and limited RAM memory, this script analyzes only
one type of trainind data among image, masked_image, and velocity.

- This script is iteratively runned by a mother bash script (data_stats.sh)

'''
# Parse arguments
parser = argparse.ArgumentParser(description='Derives mean and standard deviation values from train data.')
parser.add_argument('--var', required=True, help='Choose which type of train data to analyze.')
args = parser.parse_args()

var = args.var

# Configure paths
data_dir = "../../data/tensors"
save_dir = "../../data/data_stats"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]
indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = indices[-1]

# Load each type of train data
dataloader = DataLoaderParallel(max_index, data_dir, FILE_NUM_ZERO_PADDING)

if (var == "image") or (var == "masked_image"):
    # Analyze image data
    mean_list = dataloader.load_mean_tensor_parallel(var)
    mean_list = np.array(mean_list)
    _mean, _std   = np.mean(mean_list), np.std(mean_list)
    
elif var == "velocity":
    # Analyze velocity data
    velocity_list   = dataloader.load_velocity_tensor_parallel()
    data            = np.array(velocity_list)
    _mean, _std = np.mean(data, axis=0), np.std(data, axis=0)

else:
    # Analyze other image data
    image_tensor_list   = dataloader.load_tensor_parallel(var)
    image               = np.array(image_tensor_list)
    image               = np.mean(np.squeeze(image), axis=(1,2))
    _mean, _std     = np.mean(image, axis=0), np.std(image, axis=0)
    
# Store files
print("save {} data ({}, {})".format(var, _mean, _std))
np.save(os.path.join(save_dir, var + '_mean.npy'), _mean)
np.save(os.path.join(save_dir, var + '_std.npy'), _std)
