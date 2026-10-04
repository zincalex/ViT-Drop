#!/usr/bin/bash
datasets=("zoolake" "cifar10" "lar" "LCZ42" "CrossD" "Pest" "InfLarynge" "Bark" "WHOI22" "Kaggle38" "ZooScan20" "ColorBG")
results_dir="results/accuracy"
speed_dir="results/speed/per_model"
output_dir="results/sdr"

for speed_file in $(find "${speed_dir}" -name "*_drop*_speed.csv" -type f | sort)
do
    filename=$(basename "$speed_file" .csv)
    model=$(echo "$filename" | sed -E 's/_drop[0-9]+_.*//')
    drop_num=$(echo "$filename" | sed -E 's/.*_drop([0-9]+)_.*/\1/')
    prune_method=$(echo "$filename" | sed -E "s/^${model}_drop${drop_num}_//; s/_speed$//")

    found_results=false
    for dataset in "${datasets[@]}"
    do
        if [ -f "${results_dir}/${dataset}/output_vision_${model}_drop${drop_num}_${prune_method}_${dataset}.out" ]; then
            found_results=true
            break
        fi
    done
    if [ "$found_results" = false ]; then
        continue
    fi

    python3 src/analysis/compute_sdr.py \
        --model "${model}" \
        --prune_method "${prune_method}" \
        --drop_num "${drop_num}" \
        --datasets "${datasets[@]}" \
        --results_dir "${results_dir}" \
        --speed_dir "${speed_dir}" \
        --output_dir "${output_dir}"
done

echo "SDR results saved in ${output_dir}/"
