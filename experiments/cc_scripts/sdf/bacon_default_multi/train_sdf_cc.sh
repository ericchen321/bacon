#!/bin/bash
#SBATCH --array=0-6
#SBATCH --time=24:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_sdf_bacon_default_multi_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/

source experiments/cc_scripts/sdf/bacon_default_multi/train_sdf_per_task_cc.sh $SLURM_ARRAY_TASK_ID
