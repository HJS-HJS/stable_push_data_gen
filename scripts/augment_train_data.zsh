#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Path where mesh files are located
CONFIG_FILE="$MODULE_DIR/config/simulate_test.yaml"
# Path where train data is located
TRAIN_DATA_DIR="$(dirname $MODULE_DIR)/data_add/tensors"

vars=("image" "masked_image" "velocity" "label")

# Iterate over the array and run the Python script with different arguments
for var in "${vars[@]}"
do
    python3 augment_train_data.py --config_file $CONFIG_FILE --train_data_dir $TRAIN_DATA_DIR --var $var
done

# python3 augment_train_data_dataloader.py --config_file $CONFIG_FILE --train_data_dir $TRAIN_DATA_DIR

# Run python to convert mesh to urdf