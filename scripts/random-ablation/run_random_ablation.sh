#!/usr/bin/bash

port="29520"
GPUs="0,1"

datasets=("lar" "Pest" "WHOI22" "Bark" "LCZ42" "CrossD")

dataset_base_dir="data"
results_prune_dir="results_prune"
output_dir="results/random_ablation"
reference_selection_dir="results/selection"

configs="dinov2:block_drop:8 dinov2:block_drop:16 \
         dinov2:layer_drop_all:8 dinov2:layer_drop_all:16 \
         dinov2:layer_drop_mlp:8 \
         swinv2:layer_drop_mlp:8 swinv2:layer_drop_all:8 swinv2:block_drop:8 \
         vit:layer_drop_mlp:4 vit:block_drop:4"

seeds="0 1 2"
epochs=5
lr=0.001
batch_size=32
batch_size_eval=10

export HF_HOME="/nfsd/nldei/viespolial/.cache/hf"
export HF_DATASETS_CACHE="/nfsd/nldei/viespolial/.cache/datasets"

for dataset in "${datasets[@]}"
do
    echo "========================================================================"
    echo "[RANDOM-ABLATION] ${dataset}"
    echo "========================================================================"

    CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
        src/random-ablation/random_ablation.py \
        --dataset ${dataset} \
        --dataset_base_dir ${dataset_base_dir} \
        --results_prune_dir ${results_prune_dir} \
        --output_dir ${output_dir} \
        --reference_selection_dir ${reference_selection_dir} \
        --configs ${configs} \
        --seeds ${seeds} \
        --epochs ${epochs} \
        --lr ${lr} \
        --batch_size ${batch_size} \
        --batch_size_eval ${batch_size_eval}
done

echo "[DONE] Random ablation results in ${output_dir}/<dataset>/"
