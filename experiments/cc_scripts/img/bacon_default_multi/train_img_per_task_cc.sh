#!/bin/bash

TASK_ID=$1

if [ $TASK_ID == 0 ]
then
    IMG_NAME="tokyo"
elif [ $TASK_ID == 1 ]
then
    IMG_NAME="bath"
elif [ $TASK_ID == 2 ]
then
    IMG_NAME="copenhagen"
elif [ $TASK_ID == 3 ]
then
    IMG_NAME="hamburg"
elif [ $TASK_ID == 4 ]
then
    IMG_NAME="hiroshima"
elif [ $TASK_ID == 5 ]
then
    IMG_NAME="iss"
elif [ $TASK_ID == 6 ]
then
    IMG_NAME="lake_valhalla"
elif [ $TASK_ID == 7 ]
then
    IMG_NAME="mexico_city"
elif [ $TASK_ID == 8 ]
then
    IMG_NAME="tenby"
elif [ $TASK_ID == 9 ]
then
    IMG_NAME="kangxi"
elif [ $TASK_ID == 10 ]
then
    IMG_NAME="lanting_xu"
elif [ $TASK_ID == 11 ]
then
    IMG_NAME="requiem"
elif [ $TASK_ID == 12 ]
then
    IMG_NAME="steeds_full"
elif [ $TASK_ID == 13 ]
then
    IMG_NAME="summer"
elif [ $TASK_ID == 14 ]
then
    IMG_NAME="suzhou_full"
elif [ $TASK_ID == 15 ]
then
    IMG_NAME="suzhou_a"
elif [ $TASK_ID == 16 ]
then
    IMG_NAME="suzhou_b"
elif [ $TASK_ID == 17 ]
then
    IMG_NAME="suzhou_c"
elif [ $TASK_ID == 18 ]
then
    IMG_NAME="suzhou_d"
elif [ $TASK_ID == 19 ]
then
    IMG_NAME="suzhou_e"
elif [ $TASK_ID == 20 ]
then
    IMG_NAME="tibet"
elif [ $TASK_ID == 21 ]
then
    IMG_NAME="wang"
else
    echo "Error! Task ID not recognized"
    exit 1
fi

source experiments/train_img.sh $IMG_NAME bacon_default_multi rgb