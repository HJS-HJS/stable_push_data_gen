#!/bin/zsh

# Absolute path to the directory where this script is located
SCRIPT_DIR=$(dirname $(readlink -f "$0"))
# Absolute path to the directory where module is located
MODULE_DIR=$(dirname $SCRIPT_DIR)

# Absolute path to the directory where asset is located (meshes, urdf, pusher_urdf)
ASSSET_DIR="$MODULE_DIR/assets"
# Absolute path to the directory where urdf is located
URDF_DIR="$MODULE_DIR/assets/urdf"
# Absolute path to the directory where config is located (.yaml)
CONFIG_DIR="$MODULE_DIR/config/simulate_test.yaml"

# Get the list of file names in the directory
DISH_LIST=($(ls $URDF_DIR))

# Iterate over the array and run the Python script with different files
dish_order=1
dish_start=0

for dish in "${DISH_LIST[@]}"
do
    if [ ${dish_order} -ge ${dish_start} ] ; then
        echo "#$dish_order / ${#DISH_LIST[*]}"
        python3 $SCRIPT_DIR/datagen.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR --save_results --slider_name $dish 
        # python3 $SCRIPT_DIR/datagen.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR --slider_name $dish 
    fi
    ((dish_order+=1))
done