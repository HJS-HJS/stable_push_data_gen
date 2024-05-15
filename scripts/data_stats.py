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

if var == "image":
    # Analyze image data
    print("dataloader")
    image_list        = dataloader.load_image_tensor_parallel()
    print(type(image_list))
    image             = np.array(image_list)
    print(type(image))
    image             = np.mean(np.squeeze(image), axis=(1,2))
    print(type(image))
    mu_img            = np.mean(image)
    print(mu_img)
    np.save(os.path.join(save_dir, var + '_mean.npy'), mu_img)
    std_img           = np.std(image)
    print(std_img)
    np.save(os.path.join(save_dir, var + '_std.npy'), std_img)
    
elif var == "masked_image":
    # Analyze masked_image data
    masked_image_list = dataloader.load_masked_image_tensor_parallel()
    masked_image      = np.array(masked_image_list)
    masked_image      = np.mean(np.squeeze(masked_image), axis=(1,2))
    mu_img, std_img   = np.mean(masked_image), np.std(masked_image)
    
elif var == "velocity":
    # Analyze velocity data
    velocity_list   = dataloader.load_velocity_tensor_parallel()
    data            = np.array(velocity_list)
    mu_img, std_img = np.mean(data, axis=0), np.std(data, axis=0)

else:
    # Analyze other image data
    image_tensor_list   = dataloader.load_tensor_parallel(var)
    image               = np.array(image_tensor_list)
    image               = np.mean(np.squeeze(image), axis=(1,2))
    mu_img, std_img     = np.mean(image, axis=0), np.std(image, axis=0)
    
# Store files
np.save(os.path.join(save_dir, var + '_mean.npy'), mu_img)
np.save(os.path.join(save_dir, var + '_std.npy'), std_img)
