#!/usr/bin/bash

port="21306"
GPUs="0,1"

dataset="imagenet_demo"
prune_data_type="pt"
n_calibration_samples=512
seq_len=224

prune_method="block_drop"
block_drop_method="discrete"
# Options: discrete, consecutive ----> SwinV2 architecture not suited for consecutive

drop_n=4
model_name=dinov2
# Options: dinov3_vit, vit, swinv2, deit 

model_name_or_path=facebook/dinov2-giant-imagenet1k-1-layer
#  Options: facebook/dinov3-vitl16-pretrain-lvd1689m, google/vit-base-patch16-224, microsoft/swinv2-base-patch4-window16-256, facebook/deit-base-patch16-224

folder_name="${model_name}-${prune_method}-${block_drop_method}-drop${drop_n}"
similarity_cache_file="./results_prune/cache/${model_name}-${prune_method}-${dataset}-${n_calibration_samples}samples.pt"
output_dir=./results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
  src/compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}

block_drop_method="post_dropping"
# set only_update_config to True to save the disk memory
only_update_config=False

python src/compress.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --only_update_config $only_update_config \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}