#!/bin/bash

SCENE_NAME="$1"
DATASET_NAME="$2"
BASE_CONFIG_NAME="$3"

# define paths
BASE_CONFIG_PATH="experiments/config/nerf/${BASE_CONFIG_NAME}.ini"
if [[ "$DATASET_NAME" == "blender" ]]; then
    DATA_DIR="data/nerf_synthetic/"
elif [[ "$DATASET_NAME" == "llff" ]]; then
    DATA_DIR="data/nerf_llff_data/"
else
    echo "Error! Dataset name not recognized"
    exit 1
fi

# overwrite base configs
arg_string="--config ${BASE_CONFIG_PATH} \
--experiment_name ${BASE_CONFIG_NAME}_${SCENE_NAME} \
--dataset_path ${DATA_DIR}/${SCENE_NAME}
"

# execute
set -x
python experiments/train_radiance_field.py $arg_string
