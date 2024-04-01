#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Path where mesh files are located
CONFIG_FILE="$MODULE_DIR/config/simulate_test.yaml"
# Path where train data is located
TRAIN_DATA_DIR="$MODULE_DIR/data/tensors"

# Run python to convert mesh to urdf
python3 augment_train_data.py --config_file $CONFIG_FILE --train_data_dir $TRAIN_DATA_DIR