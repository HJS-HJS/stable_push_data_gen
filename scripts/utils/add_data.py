import os
import re
import argparse
import shutil
import parmap
import multiprocessing
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description='Derives mean and standard deviation values from train data.')
parser.add_argument('--var', required=True, help='Choose which type of train data to analyze.')
args = parser.parse_args()
var = args.var

# Configure paths
PATH = os.getcwd()
data_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data/tensors"
data_add_dir = os.path.dirname(os.path.dirname(PATH)) + "/../data_add/tensors"

# List all train data indices
FILE_NUM_ZERO_PADDING = 7
file_list = os.listdir(data_dir)
file_list = [file for file in file_list if file.startswith(var)]

indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
max_index = indices[-1]

try:
    file_list = os.listdir(data_add_dir)
except:
    raise Exception("Directory Not Exist: ", data_add_dir)
file_list = [file for file in file_list if file.startswith(var)]
indices = [int(re.findall(r'\d+', file_name)[0]) for file_name in file_list]
indices = np.sort(indices)
try:
    max_add_index = indices[-1]
    min_add_index = indices[0]
    print("move files from {} to {}".format(max_add_index, min_add_index))
except:
    raise Exception("Data Not Exists")

def move_data(idx):
    old_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(min_add_index + idx)
    new_name = ("_%0" + str(FILE_NUM_ZERO_PADDING) + 'd.npy')%(max_index + idx + 1)
    try:
        if not os.path.isfile(data_add_dir, var + old_name):
            print("{} file not exist".format(old_name))
        if not os.path.isfile(data_dir, var + new_name):
            print("{} file already exists".format(new_name))

        shutil.move(os.path.join(data_add_dir, var + old_name), os.path.join(data_dir, var + new_name))
    except:
        print("cant move ", var, " {} to {}".format(old_name, new_name))

num_cores = multiprocessing.cpu_count()
parmap.map(move_data, range(max_add_index + 1 - min_add_index), pm_pbar={'desc': 'Move ' + var + ' data'}, pm_processes=num_cores, pm_chunksize=num_cores)
