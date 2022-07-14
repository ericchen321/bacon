#!/bin/bash

exp_category="$1"
logging_subdir="$2"
resolution="$3"

declare -a shape_names=(
    "at-ot"
    # "cathedral"
    "gear_shift"
    "lunar_lander"
    "ninjago_city"
    "oak_tree"
    )

for shape_name in ${shape_names[@]}; do
    source experiments/eval_sdf.sh $shape_name $logging_subdir $exp_category $resolution
done
