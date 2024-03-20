#!/bin/zsh

# Path of shell script file
SCRIPT_PATH=$(readlink -f "$0")

# Path where shell script is located
SCRIPT_DIR=${PWD}
# Path where mesh files are located
MESH_DIR="${SCRIPT_DIR}/../assets/meshes"
# Path where urdf files will be located
URDF_DIR="${SCRIPT_DIR}/../assets/urdf"

# Run python to convert mesh to urdf
python3 mesh_to_urdf.py --mesh $MESH_DIR --urdf $URDF_DIR
