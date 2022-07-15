#!/bin/bash
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=te_nerf_bacon_default_multi_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/

source experiments/cc_scripts/nerf/bacon_default_multi/eval_nerf_per_task_cc.sh $SLURM_ARRAY_TASK_ID
