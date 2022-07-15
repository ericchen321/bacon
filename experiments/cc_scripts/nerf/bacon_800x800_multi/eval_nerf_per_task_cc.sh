#!/bin/bash

TASK_ID=$1

if [ $TASK_ID == 0 ]
then
    SCENE_NAME="lego"
elif [ $TASK_ID == 1 ]
then
    SCENE_NAME="chair"
elif [ $TASK_ID == 2 ]
then
    SCENE_NAME="drums"
elif [ $TASK_ID == 3 ]
then
    SCENE_NAME="ficus"
elif [ $TASK_ID == 4 ]
then
    SCENE_NAME="hotdog"
elif [ $TASK_ID == 5 ]
then
    SCENE_NAME="materials"
elif [ $TASK_ID == 6 ]
then
    SCENE_NAME="mic"
elif [ $TASK_ID == 7 ]
then
    SCENE_NAME="ship"
else
    echo "Error! Task ID not recognized"
    exit 1
fi

source experiments/eval_nerf.sh $SCENE_NAME blender bacon_800x800_multi 800
