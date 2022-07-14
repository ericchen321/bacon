#!/bin/bash
#SBATCH --array=0-21
#SBATCH --time=10:00:00
#SBATCH --account=def-rhodin
#SBATCH --job-name=tr_img_bacon_default_multi
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=24G
module load python/3.8
module load StdEnv/2020
module load cuda/11.1.1

cd /home/gxc321/
source BaconEnv/bin/activate
cd /home/gxc321/scratch/bacon/

declare -a img_names=(
    "tokyo"
    "bath"
    "copenhagen"
    "hamburg"
    "hiroshima"
    "iss"
    "lake_valhalla"
    "mexico_city"
    "tenby"
    "kangxi"
    "lanting_xu"
    "requiem"
    "steeds_full"
    "summer"
    "suzhou_full"
    "suzhou_a"
    "suzhou_b"
    "suzhou_c"
    "suzhou_d"
    "suzhou_e"
    "tibet"
    "wang"
    )
for img_name in ${img_names[@]}; do
    source experiments/train_img.sh $SLURM_ARRAY_TASK_ID $img_name bacon_default_multi rgb
done
