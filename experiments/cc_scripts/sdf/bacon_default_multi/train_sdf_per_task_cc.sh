#!/bin/bash

TASK_ID=$1

if [ $TASK_ID == 0 ]
then
    SHAPE_NAME="at-ot"
elif [ $TASK_ID == 1 ]
then
    SHAPE_NAME="cathedral"
elif [ $TASK_ID == 2 ]
then
    SHAPE_NAME="gear_shift"
elif [ $TASK_ID == 3 ]
then
    SHAPE_NAME="lunar_lander"
elif [ $TASK_ID == 4 ]
then
    SHAPE_NAME="ninjago_city"
elif [ $TASK_ID == 5 ]
then
    SHAPE_NAME="oak_tree"
elif [ $TASK_ID == 6 ]
then
    SHAPE_NAME="thai_statue"
else
    echo "Error! Task ID not recognized"
    exit 1
fi

source experiments/train_sdf.sh $SHAPE_NAME bacon_default_multi
