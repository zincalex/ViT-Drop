#!/usr/bin/bash

port="21804"
GPUs="0,1,2,3"

model_names=("llama2-13b" "mistral") 
drop_nums=("4" "8") 
prune_methods=("block_drop" "layer_drop_attn" "layer_drop_mlp" "layer_drop_all")

tasks=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
num_fewshots=("0" "0" "0" "0" "5" "5" "5" "10" "25")

# Place Your Huggingface Token below and uncomment
#export HUGGINGFACE_TOKEN=

for model_name in "${model_names[@]}"
do
    # Download the model to a local directory. 
    git lfs install
    if [ "$model_name" == "mistral" ]; then
        git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
        mv Mistral-7B-v0.1 ./"$model_name"_model
    elif [ "$model_name" == "llama2-13b" ]; then
        git clone https://user:$HUGGINGFACE_TOKEN@huggingface.co/meta-llama/Llama-2-13b-hf
        mv Llama-2-13b-hf ./"$model_name"_model
    fi

    for prune_method in "${prune_methods[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            cfg_path=./results_prune/"$model_name"-"$prune_method"-discrete-drop"$drop_num"/checkpoint/config.json 
            cp -f "$cfg_path" ./"$model_name"_model/config.json 
            cp ./results_prune/"$model_name"-"$prune_method"-discrete-drop"$drop_num"/checkpoint/*.py ./"$model_name"_model/ 
            echo "Eval the config of:"
            echo $cfg_path

            num_tasks=${#tasks[@]}
            for ((i=0; i<$num_tasks; i++)); do
                CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval \
                    --model hf \
                    --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16" \
                    --tasks ${tasks[$i]} \
                    --num_fewshot ${num_fewshots[$i]} \
                    --batch_size 1 \
                    --output_path ./${num_fewshots[$i]}shot_${tasks[$i]}_"$model_name"_drop"$drop_num"_"$prune_method".json >> output_"$model_name"_drop"$drop_num"_"$prune_method".out
            done
        done
    done
done
