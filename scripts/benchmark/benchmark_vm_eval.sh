#!/usr/bin/bash

port="21805"
GPUs="0,1"

model_names=("dinov2" "vit" "swinv2" "dinov3_vit" "deit")
model_paths=("facebook/dinov2-giant-imagenet1k-1-layer" "google/vit-base-patch16-224" "microsoft/swinv2-base-patch4-window16-256" "facebook/dinov3-vitl16-pretrain-lvd1689m" "facebook/deit-base-patch16-224")

drop_nums=("4" "8")
prune_methods=("layer_drop_attn" "layer_drop_mlp" "block_drop" "layer_drop_all")
datasets=("imagenet-1k" "zoolake" "cifar10" "lar" "LCZ42" "CrossD" "Pest" "InfLarynge" "Bark" "WHOI22" "Kaggle38" "ZooScan20" "ColorBG")

healed_model=false

# Place Your Huggingface Token below and uncomment
#export HUGGINGFACE_TOKEN=

for i in "${!model_names[@]}"
do
    model_name="${model_names[$i]}"
    model_path="${model_paths[$i]}"
    if [ ! -d "./${model_name}_model" ]; then
        echo "Downloading model..."
        git lfs install
        git clone https://user:$HUGGINGFACE_TOKEN@huggingface.co/${model_path}
        mv $(basename ${model_path}) ./"${model_name}"_model
        echo "Model downloaded to ./${model_name}_model"
    else
        echo "Model already exists at ./${model_name}_model, skipping download"
    fi

    for prune_method in "${prune_methods[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            if [ "$drop_num" = "0" ] && [ "$prune_method" != "${prune_methods[0]}" ]; then
                echo "Skipping drop0 for ${model_name}-${prune_method} (baseline already covered by ${prune_methods[0]})"
                continue
            fi

            if [ "$healed_model" = true ]; then
                cfg_path=./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint_healed_${datasets[0]}_demo_merged/config.json
            else
                cfg_path=./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint/config.json
            fi

            if [ ! -f "$cfg_path" ]; then
                echo "Skipping drop_num=${drop_num} for ${model_name}-${prune_method}"
                continue
            fi
            
            cp -f "$cfg_path" ./"${model_name}"_model/config.json

            if [ "$healed_model" = true ]; then
                cp ./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint_healed_${datasets[0]}_demo_merged/*.py ./"${model_name}"_model/ 2>/dev/null || true
            else
                cp ./results_prune/"${model_name}"-"${prune_method}"-discrete-drop"${drop_num}"/checkpoint/*.py ./"${model_name}"_model/
            fi
            echo "Evaluating config: $cfg_path"

            for dataset in "${datasets[@]}"
            do
                echo "Running evaluation on ${dataset}"
                if [ "$healed_model" = true ]; then
                    output_file="output_vision_${model_name}_healed_drop${drop_num}_${prune_method}_${dataset}.out"
                else
                    output_file="output_vision_${model_name}_drop${drop_num}_${prune_method}_${dataset}.out"
                fi

                CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port src/vm-eval/benchmark_vision.py \
                    --model_name_or_path ./${model_name}_model \
                    --prune_method ${prune_method} \
                    --drop_num ${drop_num} \
                    --dataset ${dataset} \
                    --dataset_base_dir data \
                    --batch_size_eval 10 \
                    --finetune_head \
                    --epochs 5 \
                    --batch_size 32 \
                    --lr 0.001 \
                    --weight_decay 0.03 \
                    --output_file "$output_file"
            done
        done
    done
done
