# Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs
![MoMos overview](images/momos.png)
This is the official repository for the paper [*Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs*](https://arxiv.org/abs/2602.14896).<br>
**Authors:**
Pedram Bakhtiarifard, Tong Chen, Jonathan Wenshøj, Erik B Dam, Raghavendra Selvan <br>
**Affiliation:**
University of Copenhagen, Department of Computer Science, Machine Learning Section, SAINTS Lab <br>
[< 1 min video overview](https://pedrambakh.com/?demo=momos)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision timm wandb pyyaml pandas numpy matplotlib seaborn scikit-learn
```

Optional (only for `bdm` metric):

```bash
python -m pip install pybdm
```

## Quick Start

Run from repository root.

```bash
# Baseline
python src/run.py --model resnet20 --prefix BASELINE

# QAT
python src/run.py --model resnet20 --method qat --q 8 --prefix QAT_Q8

# MoMos
python src/run.py --model mlp --method momos --s 4 --capacity 0.01 --q 32 --force_zero --prefix MOMOS_S4_C001

# MoMos + QAT (q < 32)
python src/run.py --model mlp --method momos --s 4 --capacity 0.01 --q 8 --force_zero --prefix MOMOS_QAT

# Same run with W&B logging/artifacts enabled
python src/run.py --model mlp --method momos --s 4 --capacity 0.01 --q 8 --force_zero --prefix MOMOS_QAT --wandb --wandb_project momos --wandb_entity your_entity
```

## Quantization Behavior

- `baseline`: no quantization.
- `method=qat`: fake-quant QAT (weight parametrization + STE) is attached once before training.
- `method=momos`: one MoMos projection is applied per train epoch (after optimizer steps), globally across all trainable blocks.
- `method=momos` with `q < 32` (`momos+qat`): QAT is attached once before training, and MoMos still runs per epoch.
- In `momos+qat`, weights are not hard-projected to `q` bits after MoMos.

Default QAT exclusion tokens:

- ResNet: `['bn']`
- ViT: `['norm']`
- MLP: `[]`

## Standalone API

### MoMos only

```python
from src.quantizers import MoMos

momos = MoMos(s=4, capacity=0.01, q=32, force_zero=True)

for epoch in range(num_epochs):
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    q_stats = momos(model)  # one projection step per epoch
```

### MoMos + QAT

```python
from src import quantizers

quantizers.quantize_qat(model, {"method": "qat", "q": 8, "exclude_layers": ["norm"]})
momos = quantizers.MoMos(s=4, capacity=0.01, q=8, force_zero=True)

for epoch in range(num_epochs):
    for x, y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)  # forward uses fake-quantized view
        loss.backward()
        optimizer.step()
    q_stats = momos(model)
```

## Run Analysis

```python
from src import models
from src.utils import load_wandb_checkpoint_from_run_id

# Download checkpoint payload directly from W&B by run id.
info = load_wandb_checkpoint_from_run_id(
    run_id="abc123xy",  # run id from W&B UI
    checkpoint="best",
    entity="your_entity",   # optional if WANDB_ENTITY is set
    project="momos",        # optional if WANDB_PROJECT is set
)

# Build the matching architecture from checkpoint config.
cfg = info["checkpoint"]["config"]
model = models.get_model(
    cfg["model"],
    cfg["num_classes"],
    img_size=cfg["img_size"],
    in_channels=cfg.get("in_channels", 3),
)

# Load weights into this model instance (in-place).
model.load_state_dict(info["checkpoint"]["model"], strict=True)
model.eval()
```

`load_model_from_wandb_run_id(model, ...)` is a convenience wrapper that performs the same state-dict loading into the `model` object you pass.

## Citation

```bibtex
@misc{bakhtiarifard2026momos,
      title={Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs},
      author={Pedram Bakhtiarifard and Tong Chen and Jonathan Wenshøj and Erik B Dam and Raghavendra Selvan},
      year={2026},
      eprint={2602.14896},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.14896},
}
```

## Contact

- pba@di.ku.dk
- raghav@di.ku.dk

## Additional Details (CLI Flags)

Core:

- `--model` (required): model name (local models like `mlp`, `resnet20`, or a timm model string).
- `--config`: named preset from `src/configs.py` (dataset + optimizer/profile defaults).
- `--prefix`: string prefix for experiment naming in `logs/`.
- `--epochs`: total training epochs.
- `--patience`: early stopping patience on validation loss (`None` disables early stopping).
- `--seed`: global random seed for reproducibility.

Data/runtime:

- `--data_dir`: dataset root directory (download/cache location).
- `--val_pct`: validation fraction/percent sampled from training split.
- `--test_pct`: test fraction/percent sampled from official test split.
- `--split_seed`: seed used for train/val/test subset sampling.
- `--device auto|cuda|mps|cpu`: device selection.
- `--gpu`: sets `CUDA_VISIBLE_DEVICES` before runtime setup.
- `--num_workers`: DataLoader worker processes.
- `--prefetch_factor`: per-worker DataLoader prefetch count (used only when `num_workers > 0`).
- `--pin_memory` / `--no_pin_memory`: DataLoader pinned-memory toggle.
- `--compile`: enables `torch.compile` (CUDA only in this project).

Progress/logging:

- `--logs_dir`: base directory for run artifacts.
- `--progress none|epoch|step`: controls step-level training progress prints (epoch summary still prints).
- `--step_updates`: number of step updates to print per epoch when `--progress step`.
- `--wandb`: enables lightweight epoch-level W&B logging and end-of-run artifact upload.
- `--wandb_project`: W&B project for this run (preferred; overrides `WANDB_PROJECT`).
- `--wandb_entity`: W&B entity/team for this run (overrides `WANDB_ENTITY`).
- W&B tags include `model:*`, `dataset:*`, and `method:*` for easy dashboard filtering.
- W&B run name format is `<model>-<method>-run_<n>` (compact); full details remain in group/tags.
- Optional environment variables: `WANDB_PROJECT` (default `momos`), `WANDB_ENTITY`, and `WANDB_DIR`.
- By default, W&B local files are written to `./wandb` in the repo.
- Artifact downloads from utility helpers default to `./wandb_artifacts`.
- Use `load_model_from_wandb_run_id(...)` to restore checkpoints directly from a W&B run id/path/URL.
- W&B is intentionally minimal:
  - per epoch: logs scalar values already present in local epoch logs (`train/val`, quantization stats, extra metrics)
  - end of run: uploads one model artifact bundle (`init.pt`, `best.pt`, `final.pt`, `results.json`, and MoMos files when available)
  - no batch-level W&B logging and no code snapshot upload

Quantization:

- `--method none|baseline|qat|momos`: quantization mode.
- `--q`: QAT bit-width (`q >= 2`). For `method=momos`, `q < 32` enables MoMos+QAT (fake quantization in forward).
- `--s`: MoMos block size (required for MoMos).
- `--k`: explicit motif count.
- `--capacity`: motif ratio used to derive `k` when `k` is not provided.
- `--force_zero`: forces zero motif inclusion in MoMos.
- `--chunk_size`: memory budget in MB for chunked pairwise-distance computation (default: `4096` MB).
- `--chunk_progress`: prints chunk-level MoMos assignment progress.
- `--chunk_progress_elements`: progress print interval in processed scalar elements.
- `--method qat` cannot be combined with MoMos-only flags (`--s`, `--k`, `--capacity`, `--force_zero`, `--chunk_size`, `--chunk_progress`, `--chunk_progress_elements`)

Metrics:

- `--metrics`: comma-separated metric names from `sparsity,l2,bdm,gzip,bz2,lzma`.
- `--all_compression_metrics_binarized`: computes compression metrics on binarized weight payloads.

## Logging and Artifacts

Each run writes:

`logs/<experiment_name>/run_<n>/`

Artifacts:

- `results.json`
- `init.pt`
- `best.pt`
- `final.pt`
- `motifs_dist.json` (MoMos runs)

For MoMos runs, `best.pt` and `final.pt` are saved from post-projection epochs (not epoch 0).

`results.json` stores:

- `config`
- `epochs` (includes `epoch=0` validation-only evaluation)
- `summary` (`wall_time`, `training_time`, final and best metrics)
- `artifacts`

When `--wandb` is enabled, `results.json` also stores:

- `config.wandb` (run id/path/url and tags)
- `summary.wandb_checkpoint_artifact` (artifact reference to download checkpoints)
- `summary.wandb_run_path`, `summary.wandb_url`
