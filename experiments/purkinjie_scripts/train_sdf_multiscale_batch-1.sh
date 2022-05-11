#!/bin/bash

python train_sdf.py --config ./config/sdf/bacon_camera_multiscale.ini
python train_sdf.py --config ./config/sdf/bacon_david_multiscale.ini
#python train_sdf.py --config ./config/sdf/bacon_dragon_multiscale.ini
#python train_sdf.py --config ./config/sdf/bacon_dragon_warrior_multiscale.ini
python train_sdf.py --config ./config/sdf/bacon_engine_multiscale.ini
