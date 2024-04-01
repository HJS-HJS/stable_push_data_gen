import os
import re
import numpy as np
import matplotlib.pyplot as plt
from utils.dataloader_parallel import DataLoaderParallel

'''
Derives mean and standard deviation values from train data.

- Because of large volume of training data and limited RAM memory, this script analyzes only
one type of trainind data among image, masked_image, and velocity.

- This script is iteratively runned by a mother bash script (data_stats.sh)

'''
# Configure paths
data_dir = "../../data/tensors"
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

# random_args = np.random.choice(np.arange(max_index), size = visualize_num, replace=False)
random_args = np.random.choice(np.arange(max_index), size = visualize_num)

fig = plt.figure(figsize=(10,10))
col = int(np.ceil(np.sqrt(visualize_num)))

#     # Analyze image data
image_list          = dataloader.load_image_tensor_parallel()
images              = np.array(image_list)
masked_image_list   = dataloader.load_masked_image_tensor_parallel()
masked_images       = np.array(masked_image_list)
velocity_list       = dataloader.load_velocity_tensor_parallel()
velocities          = np.array(velocity_list)
label_list          = dataloader.load_label_tensor_parallel()
labels              = np.array(label_list)

for i in range(visualize_num):
    ax = fig.add_subplot(col,col,i+1)
    # _temp_img = images[random_args[i]]
    _temp_img = masked_images[random_args[i]]
    ax.imshow(_temp_img.reshape(-1, _temp_img.shape[-1]))
    ax.set_title(velocities[i])
    if labels[i]==1:
        ax.text(x=80, y=88, s=labels[i], fontsize=20, color='white')
    else:
        ax.text(x=80, y=88, s=labels[i], fontsize=20, color='black')
plt.show()

