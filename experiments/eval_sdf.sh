#!/bin/bash

SHAPE_NAME="$1"
LOGGING_SUBDIR="$2"
BASE_CONFIG_NAME="$3"
RESOLUTION="$4"

# define paths
BASE_CONFIG_PATH="experiments/config/sdf_render/${BASE_CONFIG_NAME}.ini"
LOG_DIR="logs/$LOGGING_SUBDIR/"

# overwrite base configs
arg_string="--config ${BASE_CONFIG_PATH} \
--logging_root $LOG_DIR \
--experiment_name ${BASE_CONFIG_NAME}_${SHAPE_NAME} \
--ckpt ${LOG_DIR}/${BASE_CONFIG_NAME}_${SHAPE_NAME}/checkpoints/model_final.pth \
--res $RESOLUTION \
"

# execute
set -x
python experiments/render_sdf.py $arg_string
