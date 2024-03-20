#!/bin/zsh

# Absolute path to this script
SCRIPT_PATH=$(readlink -f "$0")

# Absolute path to the directory where this script is located
SCRIPT_DIR=${PWD}

ASSSET_DIR="$SCRIPT_DIR/../assets"
CONFIG_DIR="$SCRIPT_DIR/../config/simulate_test.yaml"

# Get the list of file names in the directory
python3 simulate_test.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR
