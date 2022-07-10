#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_suzhou_a_multi_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/
source experiments/train_img.sh suzhou_a bacon_default_multi rgb
