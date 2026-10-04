#!/usr/bin/bash

port="21306"
GPUs="0,1,2,3"

dataset="c4_val"
prune_data_type="pt"
n_calibration_samples=256
seq_len=2048

prune_method="layer_drop"
layer_drop_method="discrete"
target_layer="attn"
drop_n=2

model_name=llama2-13b
model_name_or_path=meta-llama/Llama-2-13b-hf

folder_name="${model_name}-${prune_method}_${target_layer}-${layer_drop_method}-drop${drop_n}"
similarity_cache_file="./results_prune/cache/${model_name}-${prune_method}_${target_layer}-${dataset}-${n_calibration_samples}samples.pt"
output_dir=./results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
  src/compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --layer_drop_norm True \
  --target_layer ${target_layer} \
  --only_update_config True \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --layer_drop_method ${layer_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}


layer_drop_method="post_dropping"
# set only_update_config to True to save the disk memory
only_update_config=False

python src/compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --only_update_config $only_update_config \
  --layer_drop_norm True \
  --target_layer ${target_layer} \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --layer_drop_method ${layer_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}
