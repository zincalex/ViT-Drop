#!/usr/bin/bash
GPU="0"

model_names=("dinov2" "dinov3_vit" "swinv2" "vit")
model_paths=("facebook/dinov2-giant-imagenet1k-1-layer" "facebook/dinov3-vitl16-pretrain-lvd1689m" "microsoft/swinv2-base-patch4-window16-256" "google/vit-base-patch16-224")
image_sizes=("224" "224" "256" "224")

models_to_run=("$@")

drop_nums=(0 2 4 6 8 12 16 20 24 28 32 40)
prune_methods=("layer_drop_attn" "layer_drop_mlp" "layer_drop_all" "block_drop")
num_runs=5
batch_sizes="128"

SPEED_DIR="./results/speed/per_model"
MASTER_SPEED_CSV="./results/speed/speed_master.csv"

# Place Your Huggingface Token below and uncomment
#export HUGGINGFACE_TOKEN=

mkdir -p "${SPEED_DIR}"
for i in "${!model_names[@]}"
do
    model_name="${model_names[$i]}"
    model_path="${model_paths[$i]}"
    image_size="${image_sizes[$i]}"

    if [ ${#models_to_run[@]} -gt 0 ] && [[ ! " ${models_to_run[*]} " =~ " ${model_name} " ]]; then
        echo "[SKIP] ${model_name} not in requested models: ${models_to_run[*]}"
        continue
    fi

    if [ ! -d "./${model_name}_model_original" ]; then
        echo "[INFO] Downloading original model: ${model_name} ..."
        git lfs install
        git clone "https://user:${HUGGINGFACE_TOKEN}@huggingface.co/${model_path}"
        mv "$(basename ${model_path})" "./${model_name}_model_original"
        echo "[INFO] Saved to ./${model_name}_model_original"
    else
        echo "[INFO] Original model already exists at ./${model_name}_model_original — skipping download"
    fi

    baseline_file="${SPEED_DIR}/${model_name}_original_speed.csv"
    echo "========================================================================"
    echo "[BENCH] Baseline: ${model_name} (original) at ${image_size}x${image_size}"
    echo "========================================================================"

    CUDA_VISIBLE_DEVICES=$GPU python src/benchmark_vision_speed.py \
        --model_path "./${model_name}_model_original" \
        --batch_sizes "${batch_sizes}" \
        --image_sizes "${image_size}" \
        --num_iterations 100 \
        --num_runs ${num_runs} \
        --save_file "${baseline_file}"

    if [ ! -d "./${model_name}_model_speed" ]; then
        echo "[INFO] Creating speed working copy of model: ${model_name} ..."
        cp -r "./${model_name}_model_original" "./${model_name}_model_speed"
    fi

    for prune_method in "${prune_methods[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            if [ "${drop_num}" -eq 0 ]; then
                continue
            fi

            cfg_path="./results_prune/${model_name}-${prune_method}-discrete-drop${drop_num}/checkpoint/config.json"

            if [ ! -f "${cfg_path}" ]; then
                echo "[SKIP] Config not found: ${cfg_path}"
                continue
            fi

            echo "------------------------------------------------------------------------"
            echo "[BENCH] ${model_name} | ${prune_method} | drop${drop_num}"
            echo "------------------------------------------------------------------------"

            cp -f "${cfg_path}" "./${model_name}_model_speed/config.json"
            cp -f "./results_prune/${model_name}-${prune_method}-discrete-drop${drop_num}/checkpoint/"*.py \
                  "./${model_name}_model_speed/" 2>/dev/null || true

            save_file="${SPEED_DIR}/${model_name}_drop${drop_num}_${prune_method}_speed.csv"

            CUDA_VISIBLE_DEVICES=$GPU python src/benchmark_vision_speed.py \
                --model_path "./${model_name}_model_speed" \
                --batch_sizes "${batch_sizes}" \
                --image_sizes "${image_size}" \
                --num_iterations 100 \
                --num_runs ${num_runs} \
                --save_file "${save_file}" \
                --baseline_file "${baseline_file}"
        done
    done
done

echo "========================================================================"
echo "[CONSOLIDATE] Building master speed CSV: ${MASTER_SPEED_CSV}"
echo "========================================================================"

python src/consolidate_speed_results.py \
    --speed_dir  "${SPEED_DIR}" \
    --output     "${MASTER_SPEED_CSV}" \
    --models     dinov2 dinov3_vit swinv2 vit \
    --methods    layer_drop_attn layer_drop_mlp layer_drop_all block_drop \
    --drop_nums  0 2 4 6 8 12 16 20 24 28 32 40 \
    --batch_size 128

echo "[DONE] Master speed CSV written to: ${MASTER_SPEED_CSV}"
