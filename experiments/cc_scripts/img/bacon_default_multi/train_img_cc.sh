#!/bin/bash
#SBATCH --array=0-21
#SBATCH --time=10:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_img_bacon_default_multi_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/

source experiments/cc_scripts/img/bacon_default_multi/train_img_per_task_cc.sh $SLURM_ARRAY_TASK_ID
