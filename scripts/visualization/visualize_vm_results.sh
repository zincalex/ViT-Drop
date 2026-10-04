#!/usr/bin/env bash

model_names=("dinov2" "vit" "swinv2" "dinov3_vit" "deit")
prune_methods=("layer_drop_attn" "layer_drop_mlp" "layer_drop_all" "block_drop")
datasets=("imagenet-1k" "zoolake" "cifar10" "lar" "LCZ42" "CrossD" "Pest" "InfLarynge" "Bark" "WHOI22" "Kaggle38" "ZooScan20" "ColorBG")

results_dir="results/accuracy"
output_dir="results/plots"

for dataset in "${datasets[@]}"
do
    for prune_method in "${prune_methods[@]}"
    do
        echo "Processing: ${dataset} - ${prune_method}"
        drop_nums_found=()
        for model in "${model_names[@]}"
        do
            pattern="${results_dir}/${dataset}/output_vision_${model}_drop*_${prune_method}_${dataset}.out"
            baseline_glob="${results_dir}/${dataset}/output_vision_${model}_drop0_*_${dataset}.out"
            for file in $pattern $baseline_glob
            do
                if [ -f "$file" ]; then
                    drop_num=$(basename "$file" | sed -E 's/.*_drop([0-9]+)_.*/\1/')
                    if [[ ! " ${drop_nums_found[@]} " =~ " ${drop_num} " ]]; then
                        drop_nums_found+=("$drop_num")
                    fi
                fi
            done
        done
        
        IFS=$'\n' drop_nums_sorted=($(sort -n <<<"${drop_nums_found[*]}"))
        unset IFS
        
        if [ ${#drop_nums_sorted[@]} -eq 0 ]; then
            echo "  ⚠️  No result files found for ${dataset} - ${prune_method}"
            echo ""
            continue
        fi
        
        echo "  Found drop numbers: ${drop_nums_sorted[@]}"
        
        python3 src/visualization/visualize_benchmark_results.py \
            --dataset ${dataset} \
            --prune_method ${prune_method} \
            --model_names ${model_names[@]} \
            --drop_nums ${drop_nums_sorted[@]} \
            --results_dir ${results_dir} \
            --output_dir ${output_dir}
        
        echo ""
    done
done
