#!/bin/bash

SHAPE_NAME="$1"
BASE_CONFIG_NAME="$2"
RESOLUTION="$3"

# define paths
BASE_CONFIG_PATH="experiments/config/sdf_render/${BASE_CONFIG_NAME}.ini"
LOG_DIR="logs"

# overwrite base configs
arg_string="--config ${BASE_CONFIG_PATH} \
--experiment_name ${BASE_CONFIG_NAME}_${SHAPE_NAME} \
--ckpt ${LOG_DIR}/${BASE_CONFIG_NAME}_${SHAPE_NAME}/checkpoints/model_final.pth \
"

# execute
set -x
python experiments/render_sdf.py $arg_string
