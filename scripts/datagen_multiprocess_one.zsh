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
for dish in "${DISH_LIST[@]}"
do
    # if [ $dish = 'takeout_cup_65oz' ] ; then
    if [ $dish = 'mug_cup02' ] ; then
        sudo pkill python3
        python3 $SCRIPT_DIR/datagen.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR --save_results --slider_name $dish 
        # python3 $SCRIPT_DIR/datagen.py --config $CONFIG_DIR --asset_dir $ASSSET_DIR --slider_name $dish 
    fi
done