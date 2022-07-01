#!/bin/bash

SHAPE_NAME="$1"
BASE_CONFIG_NAME="$2"

# define paths
SHAPE_PATH="../data/our_shapes/${SHAPE_NAME}.xyz"
BASE_CONFIG_PATH="./config/sdf/${BASE_CONFIG_NAME}.ini"

# overwrite base configs
arg_string="--config ${BASE_CONFIG_PATH} \
--experiment_name ${BASE_CONFIG_NAME}_${SHAPE_NAME} \
--point_cloud_path ${SHAPE_PATH} \
"

# execute
set -x
python train_sdf.py $arg_string
