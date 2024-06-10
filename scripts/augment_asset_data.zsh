#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Absolute path to the directory where urdf is located
URDF_DIR="$MODULE_DIR/assets/urdf"

AUG_NUM=1

# Run python to convert mesh to urdf
python3 $SCRIPT_DIR/augment_asset_data.py --num $AUG_NUM --urdf_dir $URDF_DIR