#!/bin/bash

IMG_NAME="$1"
BASE_CONFIG_NAME="$2"
COLOR="$3"

# define paths
IMG_PATH="../data/2d/${IMG_NAME}.jpg"
BASE_CONFIG_PATH="./config/img/${BASE_CONFIG_NAME}.ini"

# get image size
# https://unix.stackexchange.com/questions/75635/shell-command-to-get-pixel-size-of-an-image
RES_WIDTH=`identify ${IMG_PATH} | cut -f 3 -d " " | sed s/x.*//`
RES_HEIGHT=`identify ${IMG_PATH} | cut -f 3 -d " " | sed s/.*x//`

# overwrite base configs
arg_string="--config ${BASE_CONFIG_PATH} \
--experiment_name ${BASE_CONFIG_NAME}_${IMG_NAME} \
--res_height ${RES_HEIGHT} \
--res_width ${RES_WIDTH} \
--img_fn ${IMG_PATH} \
"

# check if the image is grayscale
if [[ "$COLOR" == "grayscale" ]]; then
    arg_string="${arg_string} --grayscale"
elif [[ "$COLOR" == "rgb" ]]; then
    :
else
    echo "Error! Color not recognized"
    exit 1
fi

# execute
set -x
python train_img.py $arg_string
