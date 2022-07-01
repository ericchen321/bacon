#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_orszaghaz_mono_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/experiments/
source train_img.sh orszaghaz bacon_default_mono rgb
