#!/bin/zsh

# Absolute path to this script
SCRIPT_PATH=$(readlink -f "$0")

# Absolute path to the directory where this script is located
SCRIPT_DIR=${PWD}

ASSET_DIR="$SCRIPT_DIR/../assets/urdf"
CONFIG_DIR="$SCRIPT_DIR/../config/config.yaml"

# Get the list of file names in the directory
MESH_LIST=($(ls $ASSET_DIR))

# Iterate over the array and run the Python script with different arguments
for mesh in "${MESH_LIST[@]}"
do
    if [ $mesh = 'takeout_cup_65oz' ] ; then
        sudo pkill python3
        python3 generate_train_data.py --config $CONFIG_DIR --save_results --slider_name $mesh
    fi
done