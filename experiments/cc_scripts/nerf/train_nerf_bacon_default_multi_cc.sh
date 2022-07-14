#!/bin/bash
#SBATCH --array=0-7
#SBATCH --time=120:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_nerf_bacon_default_multi_bacon
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/

declare -a scene_names=(
    "lego"
    "chair"
    "drums"
    "ficus"
    "hotdog"
    "materials"
    "mic"
    "ship"
    )
for scene_name in ${scene_names[@]}; do
    source experiments/train_nerf.sh $SLURM_ARRAY_TASK_ID \
    $scene_name blender bacon_default_multi 512
done
