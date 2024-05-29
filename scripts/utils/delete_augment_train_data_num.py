import os
import re
import parmap
import multiprocessing
import numpy as np

DELETE_NUM = 2845297

# Configure paths
PATH = os.getcwd()
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(PATH))) + "/data/tensors"
print(data_dir)

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.endswith('.npy')]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = int(indices[-1])

print('total data {}'.format(max_index))

image_list = []
image_list = []
image_list = []
image_list = []

def delete_data(idx):
    if idx <= DELETE_NUM:
        return
    name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(idx)
    try: 
        os.remove(data_dir + '/image' + name)
    except:
        # print("cant delete {}".format(idx + 1))
        pass
    try: 
        os.remove(data_dir + '/masked_image' + name)
    except:
        # print("cant delete {}".format(idx + 1))
        pass
    try:
        os.remove(data_dir + '/velocity' + name)
    except:
        pass
    try:
        os.remove(data_dir + '/label' + name)
    except:
        pass

num_cores = multiprocessing.cpu_count()
parmap.map(delete_data, range(max_index + 1), pm_pbar={'desc': 'Delete data'}, pm_processes=num_cores, pm_chunksize=num_cores)

print('delete {} from {} data {}'.format(DELETE_NUM, max_index + 1, max_index - DELETE_NUM))