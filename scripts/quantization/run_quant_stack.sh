#!/usr/bin/bash

GPU="0"

configs=(
    "Bark       dinov2     layer_drop_all  40"
    "InfLarynge dinov2     layer_drop_all  40"
    "WHOI22     dinov2     layer_drop_mlp  8"
    "cifar10    dinov2     block_drop      4"
    "lar        dinov2     layer_drop_all  24"
    "zoolake    dinov2     layer_drop_mlp  8"
    "Bark       swinv2     layer_drop_mlp  12"
    "Kaggle38   swinv2     layer_drop_attn 4"
    "LCZ42      swinv2     layer_drop_mlp  12"
    "zoolake    vit        layer_drop_mlp  4"
    "LCZ42      dinov3_vit layer_drop_all  20"
)

dataset_base_dir="data"
results_prune_dir="results_prune"
output_dir="results/quantization"

precisions="fp32,bf16,int8,int4"
quant_compute_dtype="fp16"
epochs=5
lr=0.001
batch_size=32
batch_size_eval=10
bench_batch_sizes="1,8,128"

export HF_HOME="/nfsd/nldei/viespolial/.cache/hf"
export HF_DATASETS_CACHE="/nfsd/nldei/viespolial/.cache/datasets"

for cfg in "${configs[@]}"
do
    read -r dataset arch method drop_n <<< "${cfg}"

    result_json="${output_dir}/${dataset}/quant_stack_${arch}_${method}.json"
    if [ -f "${result_json}" ]; then
        echo "[SKIP] ${dataset} | ${arch}/${method} already computed: ${result_json}"
        continue
    fi

    echo "========================================================================"
    echo "[QUANT-STACK] ${dataset} | ${arch}/${method} | drop 0 + drop ${drop_n}"
    echo "========================================================================"

    CUDA_VISIBLE_DEVICES=$GPU python src/quantization/quant_stack.py \
        --dataset ${dataset} \
        --dataset_base_dir ${dataset_base_dir} \
        --results_prune_dir ${results_prune_dir} \
        --output_dir ${output_dir} \
        --arch ${arch} \
        --method ${method} \
        --drop_ns "0,${drop_n}" \
        --precisions ${precisions} \
        --quant_compute_dtype ${quant_compute_dtype} \
        --epochs ${epochs} \
        --lr ${lr} \
        --batch_size ${batch_size} \
        --batch_size_eval ${batch_size_eval} \
        --bench_batch_sizes ${bench_batch_sizes}
done

echo "[DONE] Quantization stacking results in ${output_dir}/<dataset>/"
