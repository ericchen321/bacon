#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=trainCameraBacon
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv1/bin/activate
cd /home/gxc321/scratch/bacon/experiments/
python train_sdf.py --config ./config/sdf/bacon_camera_multiscale.ini
