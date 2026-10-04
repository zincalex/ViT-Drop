<h1 align="center">Retraining-Free Depth Pruning of Vision Transformers: A Cross Domain Study of Layer Redundancy</h1>

<p align="center">
  <strong>Alessandro Viespoli</strong>, <strong>Loris Nanni</strong><br>
  DEI, University of Padua
</p>

<p align="center">
  <em>Built on top of <a href="https://openreview.net/forum?id=1I7PCbOPfe">Uncovering the Redundancy in Transformers via a Unified Study of Layer Dropping</a> (TMLR 2026) by <a href="https://shwai-he.github.io/">Shwai He*</a>, <a href="https://s1ghhh.github.io/">Guoheng Sun*</a>, <a href="https://shenzheyu.github.io/">Zheyu Shen</a>, <a href="https://www.ang-li.com/">Ang Li</a>, University of Maryland, College Park</em>
</p>

<p align="center">
  <a href="#-whats-in-here">What's in here</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-supported-architectures">Architectures</a> •
  <a href="#-prepare-the-data">Data</a> •
  <a href="#-prune-a-vision-model">Prune</a> •
  <a href="#-benchmark-accuracy">Accuracy</a> •
  <a href="#-benchmark-speed-and-energy">Speed</a> •
  <a href="#-statistical-analysis">Statistics</a> •
  <a href="#-automated-model-selection">Model selection</a> •
  <a href="#-quantization-and-random-ablation">Quantization</a>
</p>


## 📖 What's in here

Prior work on large language models showed that deeper transformer layers often perform near-identity transformations, so a substantial fraction of model depth can be removed with little quality loss. This repository adapts the training-free, similarity-based layer dropping of [LLM-Drop](https://openreview.net/forum?id=1I7PCbOPfe) to vision transformers and measures how much depth redundancy pretrained vision backbones expose. It contains:

- **Retraining-free depth pruning** with 4 strategies (Block Drop, Attention Drop, MLP Drop and Joint Layer Drop) for flat and hierarchical vision transformers.
- **A classification benchmark** on 12 downstream datasets that keeps the backbone frozen and trains only a new linear head.
- **Speed, FLOPs and energy measurement** for every pruned variant.
- **Statistical analysis** of the accuracy differences, with difference and equivalence tests.
- **Automated model selection** that searches `architecture × strategy × drop count` for a target dataset and returns the best pruned model.
- **Quantization stacking** and a **random-drop ablation**.

The pruning machinery is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```
data/      dataset builders
scripts/   run scripts
src/       source code
```


## 🔧 Installation

```bash
conda create -n vision-drop python=3.10 -y
conda activate vision-drop

git clone https://github.com/zincalex/LLM-Vision-Drop.git
cd LLM-Vision-Drop

pip install -r requirements.txt
pip install -e .
```

Models are pulled from Hugging Face on first use. For gated repositories, authenticate first:

```bash
huggingface-cli login          # or: export HUGGINGFACE_TOKEN=<token>
```

`scripts/benchmark/benchmark_vm_eval.sh` and `scripts/benchmark/benchmark_vm_speed.sh` clone the checkpoints with `HUGGINGFACE_TOKEN`, so uncomment and set it near the top of both scripts before running them.


## 🧰 Supported architectures
Any flat, pre-norm vision transformer can be added. The paper uses 5 checkpoints:

| Key | Checkpoint | Blocks | Input |
|---|---|---|---|
| `vit` | `google/vit-base-patch16-224` | 12 | 224 |
| `deit` | `facebook/deit-base-patch16-224` | 12 | 224 |
| `dinov2` | `facebook/dinov2-giant-imagenet1k-1-layer` | 40 | 224 |
| `dinov3_vit` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 24 | 224 |
| `swinv2` | `microsoft/swinv2-base-patch4-window16-256` | 24 | 256 |

Input is the resolution produced by each checkpoint's image processor, which is used for every accuracy and speed result. DINOv2's config lists 518, but its processor crops to 224. DeiT-Base appears in the paper only in the ImageNet-1K analysis without retraining.


## 📂 Prepare the data
Each dataset lives in `data/<key>/` and provides these files:

```
data/<key>/
├── train.h5
├── val.h5
├── test.h5
```

Each `.h5` file has 2 root entries, and nothing else is read.

| Entry | Type | Contents |
|---|---|---|
| `images` | dataset **or** group | RGB `uint8`, values from 0 to 255 |
| `labels` | dataset `(N,)` `int32` | class index, 0-indexed and contiguous |

`images` comes in one of 2 layouts, and the loader detects which:

- **Fixed size**: one dataset of shape `(N, H, W, 3)`.
- **Variable size**: a group keyed by sample index as a string (`"0"`, `"1"`, ... `"N-1"`), each member `(H, W, 3)`. Sizes may differ per image.

ImageNet-1K only needs `test.h5`, since the models are evaluated with their original classification head.

<details>
<summary><strong>Builders for the 13 benchmarked datasets</strong></summary>

| Key | Paper name | Domain | Classes | Builder |
|---|---|---|---|---|
| `imagenet-1k` | ImageNet-1K | Natural images | 1000 | `python data/process_imagenet1k_dataset.py` |
| `cifar10` | CIFAR-10 | Natural images | 10 | `python data/create_h5_splits.py --dataset cifar10` |
| `LCZ42` | LCZ42 | Remote sensing, Sentinel-2 RGB | 17 | `python data/preprocess_lcz42_dataset.py`, then `python data/resplit_h5.py --dataset LCZ42` |
| `CrossD` | CrossD | Plankton, cross-instrument | 44 | `python data/process_daplankton_dataset.py --base_path <DAPlankton dir>` |
| `zoolake` | Zoolake | Lake zooplankton microscopy | 35 | `python data/create_h5_splits.py --dataset zoolake` |
| `lar` | Laryngeal | Laryngeal endoscopy tissue | 4 | `python data/process_laryngeal_dataset.py --source_dir <dataset dir>` |
| `InfLarynge` | InfLarynge | Laryngoscopy frame quality | 4 | `python data/process_mat_rgb_dataset.py --dataset InfLarynge --mat_path <file>.mat` |
| `Bark` | Bark | Tree bark texture | 23 | `python data/process_mat_rgb_dataset.py --dataset Bark --mat_path <file>.mat` |
| `Pest` | Pest | Crop pests | 10 | `python data/process_mat_rgb_dataset.py --dataset Pest --mat_path <file>.mat` |
| `ColorBG` | ColorBG | Breast cancer histopathology, tumor grade | 3 | `python data/process_mat_dataset.py --dataset ColorBG` |
| `Kaggle38` | Kaggle38 | Plankton, ISIIS | 38 | `python data/process_mat_dataset.py --dataset Kaggle38` |
| `WHOI22` | WHOI22 | Plankton, Imaging FlowCytobot | 22 | `python data/process_mat_dataset.py --dataset WHOI22` |
| `ZooScan20` | ZooScan20 | Zooplankton, ZooScan | 20 | `python data/process_mat_dataset.py --dataset ZooScan20` |

</details>


## 🚀 Prune a vision model

```bash
bash scripts/dropping/vision_block_drop.sh        # drop whole blocks (attention + MLP)

bash scripts/dropping/vision_layer_drop.sh        # drop attention OR MLP sublayers

bash scripts/dropping/vision_layer_drop_joint.sh  # drop across both sublayer types
```

Edit the variables at the top of the script to control the run:

| Variable | Meaning |
|---|---|
| `model_name` / `model_name_or_path` | architecture key and Hugging Face checkpoint (must match) |
| `drop_n` | how many sublayers or blocks to remove |
| `target_layer` | `attn` or `mlp` in `vision_layer_drop.sh` (`all` in the joint script) |
| `n_calibration_samples` | calibration set size, must divide evenly across GPUs |

Similarities are computed with one forward pass over 512 ImageNet-1K validation images shipped in `src/llmtuner/data/imagenet_demo_images/`, and cached in `results_prune/cache/`. Each run writes `results_prune/<model>-<method>-discrete-drop<n>/checkpoint/config.json`, where `<method>` is `block_drop`, `layer_drop_attn`, `layer_drop_mlp` or `layer_drop_all`. The config encodes which sublayers are skipped:

```jsonc
// attention sublayers only
{ "drop_attn_list": [25, 26, 24, 22], "drop_mlp_list": [] }

// MLP sublayers only
{ "drop_attn_list": [], "drop_mlp_list": [26, 27, 25, 24] }

// whole blocks (both lists identical)
{ "drop_attn_list": [26, 25, 24, 27], "drop_mlp_list": [26, 25, 24, 27] }
```

The benchmarks copy only this config into the model directory, so the removed modules are never instantiated when the model is loaded. To sweep drop counts, loop `drop_n` over the values you need. **Also generate a `drop0` variant** for every architecture and method. The benchmarks use it as the unpruned reference, and the model selection pipeline skips any pair that lacks one.


## 📊 Benchmark accuracy
```bash
bash scripts/benchmark/benchmark_vm_eval.sh
```

Set `model_names`, `drop_nums`, `prune_methods` and `datasets` at the top of the script. For every configuration, the script copies the pruned config into `./<model>_model`, replaces the classification head with a new linear layer, trains it for 5 epochs with the backbone frozen (AdamW, learning rate 0.001, weight decay 0.03, batch size 32) and evaluates on the test split. On ImageNet-1K, models with a 1000-class head are evaluated directly without training.

Each run writes `output_vision_<model>_drop<n>_<method>_<dataset>.out` with accuracy, macro precision, recall and F1, plus an HDF5 file with logits, predictions and labels. The analysis scripts expect the `.out` files in `results/accuracy/<dataset>/`.


## ⚡ Benchmark speed and energy
```bash
bash scripts/benchmark/benchmark_vm_speed.sh            # all architectures
bash scripts/benchmark/benchmark_vm_speed.sh dinov2     # only the listed ones
```

Throughput, latency, peak memory, GFLOPs and GPU board power are measured on random inputs at batch size 128, over 5 runs of 100 iterations, at the input resolution of each architecture. The script uses its own model copy (`./<model>_model_speed`), so it can run alongside the accuracy benchmark. It writes one CSV per configuration to `results/speed/per_model/` and a summary with the speedup over the unpruned model to `results/speed/speed_master.csv`.

```bash
python3 src/analysis/energy.py                   # energy per image and saving vs. the unpruned model
bash scripts/visualization/compute_sdr_all.sh    # Speedup Degradation Ratio, written to results/sdr/
```

Energy per image is the average board power during the timed runs divided by throughput. SDR is the relative accuracy loss averaged over the 12 downstream datasets divided by the relative throughput gain, so lower is better and negative values mean the pruned model is more accurate than the unpruned one.


## 📐 Statistical analysis
```bash
python3 src/analysis/statistical_tests.py --delta 2 --alpha 0.05
```

Reads the per-dataset accuracies in `results/accuracy/` and pairs every pruned configuration with the unpruned model of the same architecture over the 12 downstream datasets. For each configuration it reports:

1. the two-sided Wilcoxon signed-rank test, Holm-adjusted over the drop counts of each architecture and method;
2. the TOST equivalence test, implemented as 2 one-sided Wilcoxon tests against a margin of ±`delta` percentage points;
3. the Hodges-Lehmann estimate of the median accuracy difference with its 90% confidence interval;
4. the outcome: equivalent, superior, inferior or inconclusive.

This script, `energy.py` and `compute_sdr_all.sh` only read `results/` and run on CPU.


## 🤖 Automated model selection
No pruning setting wins on every dataset. The best architecture, strategy and drop count depend on the task, which leaves a few hundred candidates to try. The pipeline searches them and returns one deployment-ready model in 3 phases:

1. **Baseline**: trains the head of the unpruned model (`drop0`) of each `(architecture, method)` pair for 5 epochs on the training split and records its validation accuracy.
2. **Search**: trains the head for 5 epochs at increasing drop counts, in steps of 4. A direction is abandoned as soon as validation accuracy falls more than `early_stop_threshold` (0.05, that is 5 points) below its baseline.
3. **Deep fine-tune**: retrains the winner for 20 epochs on the training and validation splits combined, then evaluates once on the test split.

```bash
bash scripts/model-selection/run_selection.sh
```

**Prerequisites.** For each architecture you want searched: a local model directory and a `drop0` config under `results_prune/`. Architectures or method pairs missing either are skipped with a log line, not an error.

Among surviving variants the highest validation accuracy wins, and ties break toward the **larger** drop count, preferring more compression at equal quality. Everything lands in `results/selection/<dataset>/`:

| File | Contents |
|---|---|
| `selection_log_<dataset>.json` | every baseline and search result, the winner and the final test metrics |
| `best_head_<dataset>.pt` | classification head weights from the deep fine-tune |
| `logits_<dataset>.h5` | test-set logits and predictions for the winner |

The log is read back at startup, so an interrupted run resumes and skips the variants already recorded.

To compare architectures, these scripts take the best configuration of each architecture from the selection logs, and the unpruned model, through the same 20-epoch protocol and evaluate them on the test split:

```bash
CUDA_VISIBLE_DEVICES=0 python src/model-selection/evaluate_all_archs.py \
    --output_csv results/selection/all_archs_test.csv
CUDA_VISIBLE_DEVICES=0 python src/model-selection/evaluate_baseline_archs.py \
    --output_csv results/selection/all_archs_baseline_test.csv
```

Both swap configs in `./<arch>_model`, so run them one after the other and not in parallel with `benchmark_vm_eval.sh`. Use `--datasets` to restrict the run.


## 🧮 Quantization and random ablation
```bash
bash scripts/quantization/run_quant_stack.sh
```

Combines each listed pruned configuration and its unpruned model with weight-only quantization through `bitsandbytes`. The head is trained once in FP32 with the 5-epoch protocol and attached to FP32, BF16, INT8 and INT4 copies of the backbone, and throughput is measured at batch sizes 1, 8 and 128. It runs on a single GPU with plain `python`, skips configurations whose result already exists and writes `results/quantization/<dataset>/quant_stack_<arch>_<method>.json`.

```bash
bash scripts/random-ablation/run_random_ablation.sh
```

Removes the same number of modules as the similarity ranking, but chosen uniformly at random, with 3 seeds and the 5-epoch protocol on the validation split. The baseline and similarity-guided accuracies are read from the completed selection logs in `results/selection/`, so only the random variants are trained. Results go to `results/random_ablation/<dataset>/random_ablation_<dataset>.json`.


## 📜 License and acknowledgements
Released under the Apache 2.0 license, inherited from LLM-Drop. We thank the authors of [LLM-Drop](https://openreview.net/forum?id=1I7PCbOPfe) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for making their code available.


## 📬 Contact
- Alessandro Viespoli: alessandro.viespoli@studenti.unipd.it
- Loris Nanni: loris.nanni@unipd.it
