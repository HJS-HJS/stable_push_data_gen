import os
import re
import numpy as np
from dataloader_parallel import DataLoaderParallel

'''
Derives mean and standard deviation values from train data.

- Because of large volume of training data and limited RAM memory, this script analyzes only
one type of trainind data among image, masked_image, and velocity.

- This script is iteratively runned by a mother bash script (data_stats.sh)

'''
# Configure paths
data_dir = "../../../data/tensors"
visualize_num = 49

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = indices[-1]

# Load each type of train data
dataloader = DataLoaderParallel(max_index, data_dir, FILE_NUM_ZERO_PADDING)


#
label_list          = dataloader.load_label_tensor_parallel()
labels              = np.array(label_list)

print('True: {}\tFalse: {}\t Ratio: {:.2f}'.format(np.sum(labels), np.size(labels) - np.sum(labels), np.sum(labels)/np.size(labels)))

