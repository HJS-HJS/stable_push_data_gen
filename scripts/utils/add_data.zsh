#!/bin/zsh

# Train data types
# vars=("image" "masked_image" "velocity" "masked_origin_image")
vars=("image" "masked_image" "velocity" "label")

# Iterate over the array and run the Python script with different arguments
for var in "${vars[@]}"
do
    python3 add_data.py --var $var
done