#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Absolute path to the directory where dish mesh is located
MESH_DIR="$MODULE_DIR/assets/meshes"
# Absolute path to the directory where dish urdf will be located
URDF_DIR="$MODULE_DIR/assets/urdf"

# Run python to convert mesh to urdf
python3 $SCRIPT_DIR/mesh_to_urdf.py --mesh $MESH_DIR --urdf $URDF_DIR