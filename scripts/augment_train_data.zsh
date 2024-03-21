#!/bin/zsh

# Path where shell script is located
SCRIPT_DIR=${PWD}
# Path where urdf files will be located
URDF_DIR="${SCRIPT_DIR}/../assets/urdf"
# Path where mesh files are located
CONFIG_FILE="${SCRIPT_DIR}/../config/augment_train_data.yaml"
# Path where train data is located
TRAIN_DATA_DIR="${SCRIPT_DIR}/../data/tensor"

# Run python to convert mesh to urdf
python3 augment_train_data.py --urdf_dir $URDF_DIR --config_file $CONFIG_FILE --train_data_dir $TRAIN_DATA_DIR