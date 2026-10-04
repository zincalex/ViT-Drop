#!/usr/bin/bash

port="29500"
GPUs="0,1"

dataset="cifar10"     # Options: Bark, cifar10, CrossD, InfLarynge, lar, Pest, LCZ42, zoolake, ColorBG, Kaggle38, WHOI22, ZooScan20
dataset_base_dir="data"
results_prune_dir="results_prune"
output_dir="results/selection/${dataset}"

architectures="dinov2 dinov3_vit swinv2 vit"
prune_methods="block_drop layer_drop_attn layer_drop_mlp layer_drop_all"

# Early stopping: stop if accuracy drops more than this below baseline
early_stop_threshold=0.05

# Drop count step size
drop_step=4

# Evaluation batch size
batch_size_eval=10

# Baseline fine-tuning
baseline_epochs=5
baseline_lr=0.001
baseline_batch_size=32

# Search phase fine-tuning
search_epochs=5
search_lr=0.001
search_batch_size=32

# Deep fine-tuning of winner
deep_epochs=20
deep_lr=0.001
deep_batch_size=32
deep_warmup_ratio=0.1

# Environment
export HF_HOME="/nfsd/nldei/viespolial/.cache/hf"
export HF_DATASETS_CACHE="/nfsd/nldei/viespolial/.cache/datasets"

CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
    src/model-selection/pipeline.py \
    --dataset ${dataset} \
    --dataset_base_dir ${dataset_base_dir} \
    --results_prune_dir ${results_prune_dir} \
    --output_dir ${output_dir} \
    --architectures ${architectures} \
    --prune_methods ${prune_methods} \
    --early_stop_threshold ${early_stop_threshold} \
    --drop_step ${drop_step} \
    --batch_size_eval ${batch_size_eval} \
    --gpus ${GPUs} \
    --port ${port} \
    --baseline_epochs ${baseline_epochs} \
    --baseline_lr ${baseline_lr} \
    --baseline_batch_size ${baseline_batch_size} \
    --search_epochs ${search_epochs} \
    --search_lr ${search_lr} \
    --search_batch_size ${search_batch_size} \
    --deep_epochs ${deep_epochs} \
    --deep_lr ${deep_lr} \
    --deep_batch_size ${deep_batch_size} \
    --deep_warmup_ratio ${deep_warmup_ratio}
