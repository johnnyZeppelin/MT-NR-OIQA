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
fu, f01, f02, ..., f20, mos
```

- `fu`: distorted ERP image path
- `f01` to `f20`: 20 viewport paths
- `mos`: MOS score

Optional columns:

- `compression_type`: only needed when the compression label cannot be inferred from the path
- `distortion_level`: if absent, the code will derive it automatically by MOS quantile binning

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
data/CVIQ/view_ports/JPEG/326_fov15.png
```

and

```text
data/CVIQ/view_ports/326/326_fov15.png
```

For restored and degraded viewport paths, the code mirrors the relative subtree that comes **after** `view_ports/`.

Examples:

```text
source:   data/CVIQ/view_ports/JPEG/326_fov15.png
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
└── CVIQ/
    ├── 001.png
    ├── 002.png
    ├── ...
    ├── view_ports/
    │   └── ...
    ├── restored/
    │   ├── CVIQ_restored/
    │   └── view_ports_restored/
    ├── degraded/
    │   └── view_ports_degraded/
    └── metadata/
        ├── cviq_mos.csv
        └── manifest_cviq.csv
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
  --csv F:/ws/dataset/data/CVIQ/metadata/cviq_mos.csv \
  --path-prefix F:/ws/dataset \
  --output F:/ws/dataset/data/CVIQ/metadata/manifest_cviq.csv
```

You can also explicitly override where restored or degraded paths should be written in the manifest:

```bash
oiqa-build-cviq-manifest \
  --csv F:/ws/dataset/data/CVIQ/metadata/cviq_mos.csv \
  --path-prefix F:/ws/dataset \
  --global-restored-root data/CVIQ/restored/CVIQ_restored \
  --viewport-restored-root data/CVIQ/restored/view_ports_restored \
  --degraded-root data/CVIQ/degraded/view_ports_degraded \
  --output F:/ws/dataset/data/CVIQ/metadata/manifest_cviq.csv
```

If compression labels cannot be inferred from the paths, add a CSV column such as `compression_type` and pass:

```bash
--compression-column compression_type
```

### Step 2. Synthesize degraded viewports

```bash
oiqa-synthesize-degraded \
  --manifest F:/ws/dataset/data/CVIQ/metadata/manifest_cviq.csv \
  --output-root F:/ws/dataset/data/CVIQ/degraded/view_ports_degraded
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
  --split-csv F:/ws/dataset/data/CVIQ/metadata/splits/test_3407.csv
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

## Config note

The training code itself only depends on the manifest and split CSVs. Once the manifest is generated correctly, you can move the dataset anywhere by updating either:

- your `--path-prefix` when rebuilding the manifest, or
- the absolute paths already stored inside the manifest

---

## Notes on the global backbone

The project now supports two global-backbone modes:

- `simple_cnn`: dependency-light fallback for smoke tests and engineering validation
- `vmamba`: exact VMamba integration through the official `vmamba.py` implementation

For exact VMamba integration, the code supports:

- loading the official single-file implementation from a local VMamba source tree via `model.vmamba_repo_root`
- falling back to the vendored official `vmamba.py` copy shipped inside this project
- loading a local classification checkpoint via `model.global_pretrained_path`

The recommended exact-VMamba config for your current setup is provided as:

```text
configs/cviq_vmamba_tiny_s1l8.yaml
```

That config is pre-filled for:

- `VMamba-T [s1l8]`
- weight path `/workspace/MT-NR-OIQA/vssms/vssm1_tiny_0230s_ckpt_epoch_264.pth`
- VMamba source repo `/workspace/external_repos/VMamba`

If the official VMamba import fails at runtime, the code will warn and fall back to `simple_cnn` so the rest of the training / evaluation / table-export pipeline remains usable.

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


## Exact VMamba integration

The project now includes an exact-VMamba path through the official `vmamba.py` implementation.

Two usage modes are supported:

- **External source mode**: set `model.vmamba_repo_root` to a local VMamba repo directory that contains `vmamba.py`
- **Vendored mode**: if the external import fails, the project falls back to the vendored official copy at
  `src/oiqa_bpr_vmamba/third_party/vmamba_official.py`

The recommended config for your current setup is:

```text
configs/cviq_vmamba_tiny_s1l8.yaml
```

That config is pre-filled for:

- `VMamba-T [s1l8]`
- checkpoint path `/workspace/MT-NR-OIQA/vssms/vssm1_tiny_0230s_ckpt_epoch_264.pth`
- VMamba source repo `/workspace/external_repos/VMamba`

Typical exact-VMamba training command:

```bash
oiqa-train-cviq \
  --config /workspace/MT-NR-OIQA/configs/cviq_vmamba_tiny_s1l8.yaml \
  --device cuda:0
```

Typical exact-VMamba evaluation command:

```bash
oiqa-eval-cviq \
  --config /workspace/MT-NR-OIQA/configs/cviq_vmamba_tiny_s1l8.yaml \
  --checkpoint best \
  --split test \
  --evaluate-all-types \
  --device cuda:0
```

Notes:

- `timm` is still required.
- `triton` and CUDA selective-scan kernels are optional; if unavailable, the official file falls back to slower PyTorch implementations where possible.
- If the external VMamba repo import fails, the project will warn and use the vendored official copy instead.
