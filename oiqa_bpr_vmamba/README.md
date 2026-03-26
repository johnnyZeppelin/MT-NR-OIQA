# OIQA BPR-VMamba (CVIQ-first implementation)

This repository is a **CVIQ-first** reimplementation of a BPR + VMamba style OIQA system, organized as an installable Python package so cross-file imports stay stable.

```bash
pip install -e .
```

---

## What is included

- CVIQ manifest builder from your MOS CSV.
- Offline degradation synthesis for all viewports.
- BPR local branches based on restored and degraded pseudo references.
- Multi-scale local/global fusion with a BS-MSFA implementation.
- Multi-task heads for:
  - MOS regression
  - distortion-level classification
  - compression-type classification
- Training, evaluation, ablation, split-protocol, and full-benchmark entry points.
- Per-compression-type evaluation for `JPEG / AVC / HEVC / ref` plus overall metrics.
- Automatic result-table export to `CSV / Markdown / LaTeX`.

---

## CSV convention

The default CSV column names exactly match your setting:

```text
fu, f01, f02, ..., f20, mos, compression_type, distortion_level
```

- `fu`: distorted ERP image path
- `f01` to `f20`: 20 viewport paths
- `mos`: MOS score

Optional columns:

- `compression_type`: now treated as a standard CSV column by default
- `distortion_level`: now treated as a standard CSV column by default

Recommended values for the current CVIQ CSV are:

- `compression_type` in `['ref', 'AVC', 'HEVC', 'JPEG']`
- `distortion_level` in `[0, 1, 2, ..., 11]`

If you ever need backward compatibility with an older CSV that lacks these two columns, you can still pass:

```bash
--compression-column none --distortion-level-column none
```

The same convention can be reused later for `OIQA` and `OSIQA`.

---

## Path design

The code supports the path style you described: the CSV can store **relative paths beginning from `data/...`**, while you manually provide a filesystem prefix placed **before** `data`.

For example, if the CSV stores:

```text
data/CVIQ/326.png
data/CVIQ/view_ports/326/326_fov15.png
```

and you pass:

```text
--path-prefix F:/ws/dataset
```

then the resolved absolute paths become:

```text
F:/ws/dataset/data/CVIQ/326.png
F:/ws/dataset/data/CVIQ/view_ports/326/326_fov15.png
```

If your CSV already contains absolute paths, `--path-prefix` is not needed.

`--dataset-root` is still accepted as a backward-compatible alias of `--path-prefix`.

---

## Supported viewport layouts

The manifest builder supports both of these as long as the CSV points to the real files:

```text
data/view_ports/326/326_fov15.png
```

and

```text
data/CVIQ/view_ports/326/326_fov15.png
```

For restored and degraded viewport paths, the code mirrors the relative subtree that comes **after** `view_ports/`.

Examples:

```text
source:   data/view_ports/326/326_fov15.png
restored: data/CVIQ/restored/view_ports_restored/JPEG/326_fov15_r.png
degraded: data/CVIQ/degraded/view_ports_degraded/JPEG/326_fov15_d.png
```

```text
source:   data/CVIQ/view_ports/326/326_fov15.png
restored: data/CVIQ/restored/view_ports_restored/326/326_fov15_r.png
degraded: data/CVIQ/degraded/view_ports_degraded/326/326_fov15_d.png
```

For ERP images, the default restored root is:

```text
data/CVIQ/restored/CVIQ_restored/
```

so:

```text
data/CVIQ/326.png -> data/CVIQ/restored/CVIQ_restored/326_r.png
```

---

## Default CVIQ layout assumed by the builder

```text
data/
├── CVIQ/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── CVIQ_r/
│   ├── 001_r.png
│   ├── 002_r.png
│   └── ...
├── view_ports/
│   ├── 001/001_fov1.png
│   └── ...
├── view_ports_r/
│   ├── 001/001_fov1_r.png
│   └── ...
├── view_ports_d/
│   └── ...
└── cviq_mos.csv
```

---

## Install and check CLI entry points

After installation, these commands should exist:

- `oiqa-build-cviq-manifest`
- `oiqa-synthesize-degraded`
- `oiqa-train-cviq`
- `oiqa-eval-cviq`
- `oiqa-run-ablation`
- `oiqa-run-split-protocols`
- `oiqa-run-full-benchmark`

You can verify any one of them with:

```bash
oiqa-train-cviq --help
```

---

## Recommended command order

This is the **recommended end-to-end order** for real use.

### Step 1. Build the manifest

Typical command when the CSV stores relative `data/...` paths:

```bash
oiqa-build-cviq-manifest \
  --csv /workspace/MT-NR-OIQA/data/cviq_mos.csv \
  --path-prefix /workspace/MT-NR-OIQA \
  --global-restored-root data/CVIQ_r \
  --viewport-restored-root data/view_ports_r \
  --degraded-root data/view_ports_d \
  --output /workspace/MT-NR-OIQA/exp/meta/manifest_cviq.csv
```

You can also explicitly override where restored or degraded paths should be written in the manifest:

```bash
oiqa-build-cviq-manifest \
  --csv /workspace/MT-NR-OIQA/data/cviq_mos.csv \
  --path-prefix /workspace/MT-NR-OIQA \
  --global-restored-root data/CVIQ_r \
  --viewport-restored-root data/view_ports_r \
  --degraded-root data/view_ports_d \
  --output /workspace/MT-NR-OIQA/exp/meta/manifest_cviq.csv
```

The current project now assumes `compression_type` and `distortion_level` already exist in the CSV.

If you ever need to fall back to the older CSV style, use:

```bash
--compression-column none --distortion-level-column none
```

### Step 2. Synthesize degraded viewports

```bash
oiqa-synthesize-degraded \
  --manifest /workspace/MT-NR-OIQA/exp/meta/manifest_cviq.csv \
  --output-root /workspace/MT-NR-OIQA/data/view_ports_d
```

This writes degraded files to the exact degraded paths already stored in the manifest and checks that they are under `--output-root`.

### Step 3. Train the main model

```bash
oiqa-train-cviq --config configs/cviq_default.yaml
```

Useful overrides:

```bash
oiqa-train-cviq \
  --config configs/cviq_default.yaml \
  --override-output-dir runs/debug_exp \
  --epochs 50 \
  --batch-size 2 \
  --num-workers 2 \
  --device cuda:0
```

Resume from the last checkpoint in the output directory:

```bash
oiqa-train-cviq \
  --config configs/cviq_default.yaml \
  --override-output-dir runs/debug_exp \
  --resume auto
```

Advanced training controls:

```bash
oiqa-train-cviq \
  --config configs/cviq_default.yaml \
  --accumulation-steps 4 \
  --best-metric SRCC \
  --early-stopping-patience 20 \
  --max-train-batches 10 \
  --max-eval-batches 5
```

These are useful for low-VRAM debugging, selecting the checkpoint by `PLCC / SRCC / RMSE`, and shortening smoke runs before large experiments.

### Step 4. Evaluate the trained model

Evaluate one checkpoint and also export a combined summary for every compression type:

```bash
oiqa-eval-cviq \
  --config configs/cviq_default.yaml \
  --checkpoint best \
  --split test \
  --evaluate-all-types
```

Evaluate the whole split with an explicit checkpoint path:

```bash
oiqa-eval-cviq \
  --config configs/cviq_default.yaml \
  --checkpoint runs/cviq_default/best.pt \
  --split test
```

Evaluate only one compression type:

```bash
oiqa-eval-cviq \
  --config configs/cviq_default.yaml \
  --checkpoint runs/cviq_default/best.pt \
  --split test \
  --compression-type JPEG
```

Evaluate a user-specified split CSV:

```bash
oiqa-eval-cviq \
  --config configs/cviq_default.yaml \
  --checkpoint runs/cviq_default/best.pt \
  --split-csv /workspace/MT-NR-OIQA/exp/meta/splits/test_s3407_xxxxx.csv
```

### Step 5. Run ablations

```bash
oiqa-run-ablation --config configs/cviq_default.yaml
```

Run selected ablations only:

```bash
oiqa-run-ablation \
  --config configs/cviq_default.yaml \
  --ablations no_local,no_global,vit_backbone \
  --epochs 100
```

Useful options:

```bash
oiqa-run-ablation \
  --config configs/cviq_default.yaml \
  --include-baseline \
  --skip-existing
```

### Step 6. Run split-proportion experiments

```bash
oiqa-run-split-protocols --config configs/cviq_default.yaml
```

With explicit repeat count and a subset of protocols:

```bash
oiqa-run-split-protocols \
  --config configs/cviq_default.yaml \
  --protocols 50_50,80_20 \
  --repeats 5 \
  --epochs 100
```

Reuse already-finished repeats:

```bash
oiqa-run-split-protocols \
  --config configs/cviq_default.yaml \
  --protocols 50_50,80_20 \
  --repeats 5 \
  --skip-existing
```

### Step 7. Run the full benchmark bundle

```bash
oiqa-run-full-benchmark \
  --config configs/cviq_default.yaml \
  --work-dir runs/full_benchmark \
  --split-repeats 5
```

A shorter version for quick experiments:

```bash
oiqa-run-full-benchmark \
  --config configs/cviq_default.yaml \
  --epochs 50 \
  --split-repeats 5
```

This will create `main/`, `ablations/`, `split_protocols/`, plus top-level benchmark tables under the chosen work directory.

---

## What each stage writes

### Main training run

The training script saves:

- `resolved_config.yaml`
- `history.csv`
- `last.pt`, `best.pt`, periodic `epoch_XXX.pt`
- `best_metrics.json`, `last_metrics.json`
- per-epoch `epoch_XXX_summary.json`
- `val_best_aux_metrics.json`, `test_aux_metrics.json`
- `best_val_predictions.csv`
- `val_best_overall.json`, `val_best_per_type.csv`, `val_best_predictions.csv`
- `test_overall.json`, `test_per_type.csv`, `test_predictions.csv`
- `run_summary.json`

### Single evaluation run

Each evaluation prefix writes a compact summary bundle in three formats:

- `*_summary.csv`
- `*_summary.md`
- `*_summary.tex`

If `--evaluate-all-types` is enabled, it also writes:

- `*_multi_eval_summary.csv`
- `*_multi_eval_summary.md`
- `*_multi_eval_summary.tex`

### Ablation output

Each ablation gets its own subdirectory and `train.log`.

Aggregated files include:

- `ablation_summary.csv`
- `ablation_table.csv`
- `ablation_table.md`
- `ablation_table.tex`

If a baseline run is present, baseline-relative deltas are included.

### Split-protocol output

Per-protocol outputs include:

- `per_repeat.csv`
- `average.json`
- `per_type_average.csv`

Global aggregated files include:

- `all_protocols_per_repeat.csv`
- `all_protocols_average.csv`
- `split_protocol_table.csv`
- `split_protocol_table.md`
- `split_protocol_table.tex`
- `split_protocol_per_type_table.csv`
- `split_protocol_per_type_table.md`
- `split_protocol_per_type_table.tex`

### Full benchmark output

Derived tables include:

- `main/main_result_table.{csv,md,tex}`
- `main/main_per_type_table.{csv,md,tex}`
- `ablations/ablation_table.{csv,md,tex}`
- `split_protocols/split_protocol_table.{csv,md,tex}`
- `split_protocols/split_protocol_per_type_table.{csv,md,tex}`
- `benchmark_table.{csv,md,tex}`
- `benchmark_summary.json`

---

## Current CVIQ label settings

For your current CSV, the project now assumes:

- `model.num_distortion_levels = 12`
- `model.compression_classes = [ref, AVC, HEVC, JPEG]`

This matches:

- `distortion_level` in `[0, 1, ..., 11]`
- `compression_type` in `['ref', 'AVC', 'HEVC', 'JPEG']`

## Config note

The training code itself only depends on the manifest and split CSVs. Once the manifest is generated correctly, you can move the dataset anywhere by updating either:

- your `--path-prefix` when rebuilding the manifest, or
- the absolute paths already stored inside the manifest

---

## Notes on the global backbone

The paper uses VMamba for global feature extraction. This code keeps that interface, but the actual backbone is wrapped so you can:

- use an installable VMamba/timm backbone when available
- switch to a ViT-like backbone for backbone ablations
- keep the rest of the code unchanged

This also keeps the project easy to extend to OIQA later by only resizing ERP inputs to `4096x2048` before the global branch.

---

## Engineering notes

- The package can be imported either after `pip install -e .` or directly in tests because `tests/conftest.py` injects `src/` onto `PYTHONPATH`.
- Degraded viewport synthesis uses a stable hash instead of Python's process-randomized `hash()`, so offline and online degradation seeds stay reproducible across runs and machines.
- `oiqa-synthesize-degraded` writes files to the degraded paths already stored in the manifest and validates that those destinations live under `--output-root`, so the generated files and the training manifest stay aligned.
- When `timm`, `torchvision`, or an exact VMamba backbone is unavailable, the project falls back automatically to an internal multi-scale CNN backbone; that fallback internally downsamples very large ERP inputs before feature extraction to reduce OOM risk.
- Compression-type filtering happens at dataset level, so `oiqa-eval-cviq --compression-type JPEG` uses the same manifest and split files without rebuilding them.

---

## Minimal practical recommendation

For a clean first run, use this exact order:

1. `oiqa-build-cviq-manifest`
2. `oiqa-synthesize-degraded`
3. `oiqa-train-cviq`
4. `oiqa-eval-cviq --checkpoint best --split test --evaluate-all-types`
5. `oiqa-run-ablation`
6. `oiqa-run-split-protocols`
7. `oiqa-run-full-benchmark`

If something goes wrong, check CLI help first:

```bash
oiqa-run-full-benchmark --help
```
