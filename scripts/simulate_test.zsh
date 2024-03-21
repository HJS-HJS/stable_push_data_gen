#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Absolute path to the directory where asset is located (meshes, urdf, pusher_urdf)
ASSSET_DIR="$MODULE_DIR/assets"
# Absolute path to the directory where config is located (.yaml)
CONFIG_DIR="$MODULE_DIR/config/simulate_test.yaml"

# Get the list of file names in the directory
python3 $SCRIPT_DIR/simulate_test.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR
