import json
import os
import random
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import torch
import numpy as np


def normalize_pct(value, field_name):
    """Normalize split value to a fraction in ``(0, 1]``.

    Args:
        value: Ratio in ``(0, 1]``, percent in ``(0, 100]``, or ``None``.
        field_name: Field label used in error messages (e.g. ``val_pct``).

    Returns:
        Normalized fraction, or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in (0,1] or (0,100]. Got {value}")
    return value


def seed_all(seed):
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Integer seed used across libraries.
    """
    seed = int(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode):
    """Resolve a device mode string to a concrete runtime device.

    Args:
        mode: One of ``auto``, ``cuda``, ``mps``, or ``cpu``.

    Returns:
        Resolved device string.
    """
    mode = str(mode).lower()
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if has_mps else "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return "cuda"
    if mode == "mps":
        if not has_mps:
            raise RuntimeError("Requested MPS but it is not available.")
        return "mps"
    if mode == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device mode: {mode}")


def runtime_profile(device, num_workers=None, pin_memory=None, prefetch_factor=None):
    """Return default data-loading/runtime settings for a device.

    Args:
        device: Resolved device string from ``resolve_device``.
        num_workers: Optional explicit worker override.
        pin_memory: Optional explicit pin-memory override.
        prefetch_factor: Optional DataLoader prefetch override (workers > 0 only).

    Returns:
        Runtime config dict used by data loading and transfers.
    """
    profiles = {
        "cuda": {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "transfer": {"non_blocking": True, "channels_last": True},
        },
        "mps": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "transfer": {"non_blocking": False, "channels_last": False},
        },
        "cpu": {
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "transfer": {"non_blocking": False, "channels_last": False},
        },
    }

    cfg = dict(profiles[device])
    cfg["transfer"] = dict(cfg["transfer"])

    if num_workers is not None:
        cfg["num_workers"] = int(num_workers)
    if pin_memory is not None:
        cfg["pin_memory"] = bool(pin_memory)

    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)
        if prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be > 0, got {prefetch_factor}")
        cfg["prefetch_factor"] = prefetch_factor

    if cfg["num_workers"] <= 0:
        cfg["num_workers"] = 0
        cfg["persistent_workers"] = False
        cfg["prefetch_factor"] = None
    return cfg


def configure_cuda_fast_path():
    """Enable common CUDA fast-path settings when CUDA is available."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def format_seconds(seconds):
    """Format seconds as ``MM:SS`` or ``HH:MM:SS``.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable duration string.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def next_run_dir(logs_dir, exp_name):
    """Create and return the next unused run directory.

    Args:
        logs_dir: Base logs directory.
        exp_name: Experiment-name subdirectory.

    Returns:
        ``pathlib.Path`` pointing to the created run directory.
    """
    exp_dir = Path(logs_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        run_dir = exp_dir / f"run_{idx}"
        try:
            run_dir.mkdir(exist_ok=False)
            return run_dir
        except FileExistsError:
            idx += 1


def make_experiment_name(model, dataset, quantization=None, prefix=""):
    """Build a stable experiment name from run configuration.

    Args:
        model: Model identifier.
        dataset: Dataset identifier.
        quantization: Optional quantization config dict.
        prefix: Optional experiment prefix.

    Returns:
        String experiment name used for log directory grouping.
    """
    parts = [prefix] if prefix else []
    parts += [model, dataset]
    if quantization:
        method = str(quantization.get("method", "qat")).lower()
        if method == "qat":
            parts.append(f"qat_q{quantization.get('q', 32)}")
        elif method == "momos":
            z = "z1" if quantization.get("force_zero", False) else "z0"
            parts.append(f"momos_s{quantization.get('s')}_k{quantization.get('k')}_q{quantization.get('q', 32)}_{z}")
        else:
            if quantization.get("q") is not None:
                parts.append(f"{method}_q{quantization.get('q')}")
            else:
                parts.append(method)
    return "_".join(str(part) for part in parts if part not in [None, ""])


def print_run_header(cfg, split_info, exp_name, run_dir):
    """Print a compact summary of run settings and split details.

    Args:
        cfg: Resolved run configuration dict.
        split_info: Split metadata produced by ``get_dataloaders``.
        exp_name: Experiment name string.
        run_dir: Filesystem path for this run.
    """
    print(f"Run: exp={exp_name} | dir={run_dir}")
    model_params = cfg.get("model_num_params")
    if model_params is not None:
        try:
            model_params = int(model_params)
            model_params_m = model_params / 1_000_000.0
            params_label = f"{model_params_m:.3f}M ({model_params:,})"
        except (TypeError, ValueError):
            params_label = str(model_params)
    else:
        params_label = "-"
    print(f"Model: name={cfg['model']} | params={params_label}")
    print(f"Data: dataset={cfg['dataset']} | device={cfg['device']} | seed={cfg['seed']}")
    optimizer = cfg.get("optimizer")
    lr = cfg.get("learning_rate")
    weight_decay = cfg.get("weight_decay")
    scheduler = cfg.get("lr_scheduler", "none")
    batch_size = cfg.get("batch_size")
    epochs = cfg.get("epochs")
    patience = cfg.get("patience")
    momentum = cfg.get("momentum") if str(optimizer).lower() == "sgd" else None

    train_parts = [
        f"optimizer={optimizer}",
        f"lr={float(lr):.3e}" if lr is not None else "lr=-",
        f"weight_decay={float(weight_decay):.3e}" if weight_decay is not None else "weight_decay=-",
        f"scheduler={scheduler}",
        f"batch_size={batch_size}",
        f"epochs={epochs}",
        f"patience={patience if patience is not None else 'none'}",
    ]
    if momentum is not None:
        train_parts.append(f"momentum={float(momentum):.3f}")
    print("Train: " + " | ".join(train_parts))

    runtime = cfg.get("runtime", {})
    if runtime:
        print(
            "Runtime: "
            f"workers={runtime.get('num_workers')} | "
            f"pin_memory={runtime.get('pin_memory')} | "
            f"persistent_workers={runtime.get('persistent_workers')} | "
            f"prefetch_factor={runtime.get('prefetch_factor')}"
        )
    progress = cfg.get("progress")
    if progress:
        print(f"Progress: mode={progress.get('mode')} | step_updates={progress.get('step_updates')}")

    if split_info.get("has_proper_test", False):
        print(
            f"split_mode={split_info['split_mode']} | "
            f"split(train/val/test)={split_info['train_size']}/{split_info['val_size']}/{split_info['test_size']}"
        )
    else:
        print(
            f"split_mode={split_info['split_mode']} | "
            f"split(train/val)={split_info['train_size']}/{split_info['val_size']} | test=none"
        )

    method_label, args_label = quantization_overview(cfg.get("quantization"))
    if args_label == "-":
        print(f"Method: {method_label}")
    else:
        print(f"Method: {method_label} | {args_label}")


def load_runs(logs_dir="logs", completed_only=False, model=None, dataset=None, show_summary=False):
    """Load run JSON logs into memory.

    Args:
        logs_dir: Base logs directory to scan.
        completed_only: If True, keep only completed runs.
        model: Optional exact model filter applied while loading.
        dataset: Optional exact dataset filter applied while loading.
            This is a convenience; the same filters are also available in ``filter_runs``.
        show_summary: If True, print grouped run overview using ``run_summary``.

    Returns:
        List of run dictionaries.
    """
    runs = []
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return runs

    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            path = run_dir / "results.json"
            if not path.exists():
                continue

            try:
                with open(path, "r") as f:
                    run = json.load(f)
            except Exception:
                continue

            if completed_only and not run.get("summary", {}).get("completed", False):
                continue
            cfg = run.get("config", {})
            if model is not None and cfg.get("model") != model:
                continue
            if dataset is not None and cfg.get("dataset") != dataset:
                continue

            run["_exp_name"] = exp_dir.name
            run["_run_dir"] = str(run_dir)
            run["_results_path"] = str(path)
            runs.append(run)

    if show_summary:
        run_summary(runs)
    return runs


def quantization_overview(qcfg):
    """Build method/args strings for summary tables.

    Args:
        qcfg: Quantization config dictionary or ``None``.

    Returns:
        Tuple ``(method_label, args_label)``.
    """
    if not qcfg:
        return "baseline", "-"

    method = str(qcfg.get("method", "qat")).lower()

    if method == "qat":
        q = qcfg.get("q", 32)
        return "qat", f"q={q}"

    if method == "momos":
        q = qcfg.get("q", 32)
        method_label = "momos+qat" if q is not None and int(q) < 32 else "momos"
        parts = []
        blocks = qcfg.get("total_num_blocks")
        chunk_size = qcfg.get("chunk_size")
        ordered = [
            ("q", "q"),
            ("s", "s"),
            ("k", "k"),
            ("capacity", "capacity"),
            ("total_num_blocks", "blocks"),
            ("force_zero", "force_zero"),
            ("chunk_size", "chunk_size"),
            ("chunk_progress_elements", "chunk_progress_elements"),
        ]
        for key, label in ordered:
            if key == "total_num_blocks":
                value = blocks
            elif key == "chunk_size":
                value = chunk_size
            else:
                value = qcfg.get(key)
            if value is not None:
                if key == "chunk_size":
                    parts.append(f"{label}={value}MB")
                else:
                    parts.append(f"{label}={value}")
        return method_label, ", ".join(parts) if parts else "-"

    parts = []
    for key in (
        "q",
        "s",
        "k",
        "capacity",
        "force_zero",
        "chunk_size",
        "chunk_progress_elements",
    ):
        value = qcfg.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return method, ", ".join(parts) if parts else "-"


def run_summary(runs):
    """Print and return grouped run counts for quick overview.

    Args:
        runs: List of run dictionaries (for example from ``load_runs``).

    Returns:
        List of grouped rows with keys:
        ``model``, ``dataset``, ``method``, ``args``, ``runs``, ``avg_time``.
        ``avg_time`` is average wall time in minutes (or ``None`` when unavailable).
    """
    grouped = defaultdict(lambda: {"runs": 0, "wall_time_sum": 0.0, "wall_time_count": 0})
    for run in runs:
        cfg = run.get("config", {})
        model = cfg.get("model", "<unknown>")
        dataset = cfg.get("dataset", "<unknown>")
        method, args = quantization_overview(cfg.get("quantization"))
        key = (str(model), str(dataset), str(method), str(args))
        grouped[key]["runs"] += 1

        wall_time = run.get("summary", {}).get("wall_time")
        if wall_time is not None:
            try:
                wall_time = float(wall_time)
                if wall_time >= 0.0:
                    grouped[key]["wall_time_sum"] += wall_time
                    grouped[key]["wall_time_count"] += 1
            except (TypeError, ValueError):
                pass

    rows = [
        {
            "model": key[0],
            "dataset": key[1],
            "method": key[2],
            "args": key[3],
            "runs": int(stats["runs"]),
            "avg_time": (
                (float(stats["wall_time_sum"]) / float(stats["wall_time_count"])) / 60.0
                if int(stats["wall_time_count"]) > 0
                else None
            ),
        }
        for key, stats in grouped.items()
    ]
    rows.sort(key=lambda row: (row["model"], row["dataset"], row["method"], row["args"]))

    if not rows:
        print("No runs found.")
        return rows

    headers = ["model", "dataset", "method", "args", "runs", "avg_time"]

    def display_value(row, header):
        if header == "avg_time":
            value = row.get("avg_time")
            if value is None:
                return "-"
            return f"{float(value):.2f}m"
        return str(row[header])

    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(display_value(row, header)))

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep_line = "-+-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["model"]).ljust(widths["model"]),
                    str(row["dataset"]).ljust(widths["dataset"]),
                    str(row["method"]).ljust(widths["method"]),
                    str(row["args"]).ljust(widths["args"]),
                    str(row["runs"]).rjust(widths["runs"]),
                    display_value(row, "avg_time").rjust(widths["avg_time"]),
                ]
            )
        )
    return rows


def _coerce_float(value):
    """Convert numeric-like values to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _final_epoch_row(run):
    """Return the row corresponding to the final logged epoch."""
    epochs = run.get("epochs", [])
    if not epochs:
        return {}

    summary = run.get("summary", {})
    final_epoch = summary.get("final_epoch")
    try:
        final_epoch = int(final_epoch) if final_epoch is not None else None
    except (TypeError, ValueError):
        final_epoch = None

    if final_epoch is not None:
        for row in reversed(epochs):
            try:
                if int(row.get("epoch")) == final_epoch:
                    return row
            except (TypeError, ValueError):
                continue

    best = None
    for row in epochs:
        try:
            epoch_idx = int(row.get("epoch"))
        except (TypeError, ValueError):
            continue
        if best is None or epoch_idx > best[0]:
            best = (epoch_idx, row)
    return best[1] if best is not None else epochs[-1]


def _final_metrics_from_run(run):
    """Extract final (not best) scalar metrics from one run log."""
    summary = run.get("summary", {})
    final_row = _final_epoch_row(run)
    row_metrics = final_row.get("metrics", {}) if isinstance(final_row, dict) else {}

    val_acc = _coerce_float(summary.get("final_val_acc"))
    if val_acc is None:
        val_acc = _coerce_float(final_row.get("val_acc") if isinstance(final_row, dict) else None)

    out = {
        "val_acc": val_acc,
        "test_acc": _coerce_float(summary.get("test_acc")),
        "sparsity": _coerce_float(row_metrics.get("sparsity")),
        "weight_l2": _coerce_float(row_metrics.get("weight_l2")),
        "bdm_complexity": _coerce_float(row_metrics.get("bdm_complexity")),
        "gzip_compression_rate": _coerce_float(row_metrics.get("gzip_compression_rate")),
        "bz2_compression_rate": _coerce_float(row_metrics.get("bz2_compression_rate")),
        "lzma_compression_rate": _coerce_float(row_metrics.get("lzma_compression_rate")),
    }
    return out


def _resolve_metric_columns(metrics):
    """Resolve user-facing metric names to canonical table columns."""
    aliases = {
        "val_acc": "val_acc",
        "test_acc": "test_acc",
        "sparsity": "sparsity",
        "l2": "weight_l2",
        "weight_l2": "weight_l2",
        "bdm": "bdm_complexity",
        "bdm_complexity": "bdm_complexity",
        "gzip": "gzip_compression_rate",
        "gzip_compression_rate": "gzip_compression_rate",
        "bz2": "bz2_compression_rate",
        "bz2_compression_rate": "bz2_compression_rate",
        "lzma": "lzma_compression_rate",
        "lzma_compression_rate": "lzma_compression_rate",
        "lzma2": "lzma_compression_rate",
        "lza2": "lzma_compression_rate",
    }
    default_cols = [
        "val_acc",
        "test_acc",
        "sparsity",
        "weight_l2",
        "bdm_complexity",
        "gzip_compression_rate",
        "bz2_compression_rate",
        "lzma_compression_rate",
    ]

    if metrics is None:
        return default_cols

    resolved = []
    for name in metrics:
        key = aliases.get(str(name).strip().lower())
        if key is None:
            known = ", ".join(sorted(aliases.keys()))
            raise ValueError(f"Unknown metric '{name}'. Known aliases: {known}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def _metric_display_name(canonical_name):
    """Map canonical metric key to compact table column name."""
    mapping = {
        "weight_l2": "l2",
        "bdm_complexity": "bdm",
        "gzip_compression_rate": "gzip",
        "bz2_compression_rate": "bz2",
        "lzma_compression_rate": "lzma",
    }
    return mapping.get(canonical_name, canonical_name)


def _final_results_rows(
    runs,
    model,
    dataset,
    include_std=False,
    metrics=None,
    completed=True,
):
    """Build grouped final-results rows for one model+dataset.

    Args:
        runs: List returned by ``load_runs``.
        model: Exact model name to include.
        dataset: Exact dataset name to include.
        include_std: If True, add ``<metric>_std`` columns.
        metrics: Optional list of metric names/aliases to include.
            Defaults to val/test acc + common compression metrics.
        completed: If True, include only completed runs.

    Returns:
        Tuple ``(rows, headers)`` where ``rows`` is a list of dicts and
        ``headers`` is the preferred column order.
    """
    selected = filter_runs(runs, model=model, dataset=dataset, completed=completed)
    metric_cols = _resolve_metric_columns(metrics)
    metric_names = [_metric_display_name(col) for col in metric_cols]
    grouped = defaultdict(list)

    for run in selected:
        cfg = run.get("config", {})
        method, args = quantization_overview(cfg.get("quantization"))
        grouped[(str(method), str(args))].append(_final_metrics_from_run(run))

    rows = []
    for (method, args), metrics_list in grouped.items():
        row = {
            "model": str(model),
            "dataset": str(dataset),
            "method": method,
            "args": args,
            "runs": int(len(metrics_list)),
        }

        for col, name in zip(metric_cols, metric_names):
            vals = [m.get(col) for m in metrics_list if m.get(col) is not None]
            row[name] = float(np.mean(vals)) if vals else None
            if include_std:
                row[f"{name}_std"] = float(np.std(vals)) if vals else None

        rows.append(row)

    headers = ["model", "dataset", "method", "args", "runs"] + metric_names
    if include_std:
        headers += [f"{name}_std" for name in metric_names]
    rows.sort(key=lambda r: (r["method"], r["args"]))
    return rows, headers


def runs_df(
    runs,
    model,
    dataset,
    include_std=False,
    metrics=None,
    completed=True,
):
    """Return grouped final-results table as a pandas DataFrame.

    Args:
        runs: List returned by ``load_runs``.
        model: Exact model name to include.
        dataset: Exact dataset name to include.
        include_std: If True, add ``<metric>_std`` columns.
        metrics: Optional list of metric names/aliases to include.
            Defaults to val/test acc + common compression metrics.
        completed: If True, include only completed runs.

    Returns:
        ``pandas.DataFrame`` with one row per method/config group.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("runs_df requires pandas to be installed") from exc

    rows, headers = _final_results_rows(
        runs=runs,
        model=model,
        dataset=dataset,
        include_std=include_std,
        metrics=metrics,
        completed=completed,
    )
    if not rows:
        return pd.DataFrame(columns=headers)

    df = pd.DataFrame(rows)
    for col in headers:
        if col not in df.columns:
            df[col] = None
    return df[headers].reset_index(drop=True)


def final_results_table(
    runs,
    model,
    dataset,
    include_std=False,
    metrics=None,
    completed=True,
):
    """Compatibility wrapper: print and return grouped final-results rows.

    Prefer using ``runs_df(...)`` for analysis/export.
    """
    df = runs_df(
        runs=runs,
        model=model,
        dataset=dataset,
        include_std=include_std,
        metrics=metrics,
        completed=completed,
    )
    if df.empty:
        print(f"No runs found for model={model!r}, dataset={dataset!r}.")
        return []
    print(df.to_string(index=False))
    return df.to_dict(orient="records")


def filter_runs(
    runs,
    model=None,
    dataset=None,
    completed=None,
    quantized=None,
    method=None,
    force_zero=None,
    q=None,
    s=None,
    k=None,
    capacity=None,
    where=None,
):
    """Filter run dictionaries by config/summary fields.

    Args:
        runs: List returned by ``load_runs``.
        model: Optional exact model filter.
        dataset: Optional exact dataset filter.
        completed: Optional completion-status filter.
        quantized: Optional flag filter for quantized/non-quantized.
        method: Optional quantization method filter. Supports ``baseline``,
            ``none``, ``qat``, ``momos``, and ``momos+qat`` (alias:
            ``momos_qat``).
        force_zero: Optional MoMos ``force_zero`` filter.
        q: Optional bit-width filter.
        s: Optional MoMos block-size filter.
        k: Optional MoMos motif-count filter.
        capacity: Optional MoMos capacity filter.
        where: Optional callable predicate ``where(run) -> bool``.

    Returns:
        Filtered list of run dictionaries.
    """
    method_filter = None
    if method is not None:
        method_filter = str(method).lower()

    out = []
    for run in runs:
        cfg = run.get("config", {})
        summary = run.get("summary", {})
        qcfg = cfg.get("quantization")
        q_method = None
        if qcfg is not None:
            q_method = str(qcfg.get("method", "qat")).lower()

        if model is not None and cfg.get("model") != model:
            continue
        if dataset is not None and cfg.get("dataset") != dataset:
            continue
        if completed is not None and bool(summary.get("completed", False)) != bool(completed):
            continue
        if quantized is not None and (qcfg is not None) != bool(quantized):
            continue
        if method_filter is not None:
            if method_filter in {"baseline", "none"}:
                if qcfg is not None:
                    continue
            elif method_filter in {"momos+qat", "momos_qat", "momosqat"}:
                if q_method != "momos":
                    continue
                if int(qcfg.get("q", 32)) >= 32:
                    continue
            elif q_method != method_filter:
                continue
        if force_zero is not None:
            if not qcfg or bool(qcfg.get("force_zero", False)) != bool(force_zero):
                continue
        if q is not None:
            if not qcfg or int(qcfg.get("q", -1)) != int(q):
                continue
        if s is not None:
            if not qcfg or int(qcfg.get("s", -1)) != int(s):
                continue
        if k is not None:
            if not qcfg or int(qcfg.get("k", -1)) != int(k):
                continue
        if capacity is not None:
            if not qcfg or qcfg.get("capacity") is None:
                continue
            if abs(float(qcfg.get("capacity")) - float(capacity)) > 1e-12:
                continue
        if where is not None and not where(run):
            continue

        out.append(run)

    return out


def load_wandb_checkpoint_from_results(
    results_path,
    checkpoint="best",
    artifact_ref=None,
    map_location="cpu",
    download_root=None,
):
    """Load a checkpoint from W&B using artifact info in local ``results.json``."""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    with open(results_path, "r") as f:
        run = json.load(f)

    if artifact_ref is None:
        artifact_ref = run.get("summary", {}).get("wandb_checkpoint_artifact")
    if artifact_ref is None:
        wb_cfg = run.get("config", {}).get("wandb", {})
        if wb_cfg.get("enabled") and wb_cfg.get("artifact_name") and wb_cfg.get("project"):
            entity = wb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
            if entity:
                artifact_ref = f"{entity}/{wb_cfg['project']}/{wb_cfg['artifact_name']}:latest"
    if artifact_ref is None:
        raise ValueError(
            "No W&B artifact reference found. "
            "Run with --wandb enabled, or pass artifact_ref explicitly."
        )

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError("load_wandb_checkpoint_from_results requires wandb package") from exc

    if download_root is None:
        repo_root = Path(__file__).resolve().parents[1]
        download_root = repo_root / "wandb_artifacts"

    api = wandb.Api()
    artifact = api.artifact(str(artifact_ref))
    artifact_dir = Path(artifact.download(root=str(download_root)))

    ckpt_name = str(checkpoint)
    if not ckpt_name.endswith(".pt"):
        ckpt_name = f"{ckpt_name}.pt"
    ckpt_path = artifact_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint '{ckpt_name}' not found in artifact {artifact_ref}")

    payload = torch.load(ckpt_path, map_location=map_location)
    return {
        "checkpoint": payload,
        "checkpoint_path": str(ckpt_path),
        "artifact_ref": str(artifact_ref),
        "artifact_dir": str(artifact_dir),
    }


def load_model_from_wandb_results(
    model,
    results_path,
    checkpoint="best",
    artifact_ref=None,
    map_location="cpu",
    download_root=None,
    strict=True,
):
    """Load W&B checkpoint weights into the provided model using local results metadata."""
    out = load_wandb_checkpoint_from_results(
        results_path=results_path,
        checkpoint=checkpoint,
        artifact_ref=artifact_ref,
        map_location=map_location,
        download_root=download_root,
    )
    payload = out["checkpoint"]
    state = payload.get("model", payload)
    load_info = model.load_state_dict(state, strict=strict)
    out["missing_keys"] = list(getattr(load_info, "missing_keys", []))
    out["unexpected_keys"] = list(getattr(load_info, "unexpected_keys", []))
    return out


def _resolve_wandb_run_path(run_id, entity=None, project=None):
    """Resolve run selector to canonical ``entity/project/run_id`` path."""
    run_id = str(run_id).strip()
    if run_id == "":
        raise ValueError("run_id must be a non-empty string")

    # Accept full website URLs like:
    # https://wandb.ai/<entity>/<project>/runs/<run_id>
    if run_id.startswith("http://") or run_id.startswith("https://"):
        parsed = urlparse(run_id)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 4 and parts[2] == "runs":
            return f"{parts[0]}/{parts[1]}/{parts[3]}"
        raise ValueError(
            "Could not parse W&B run URL. Expected format "
            "'https://wandb.ai/<entity>/<project>/runs/<run_id>'."
        )

    slash_count = run_id.count("/")
    if slash_count == 2:
        # Already entity/project/run_id
        return run_id
    if slash_count == 1:
        # project/run_id + external entity
        entity = entity or os.getenv("WANDB_ENTITY")
        if entity is None:
            raise ValueError(
                "Run selector '<project>/<run_id>' requires entity. "
                "Pass entity=... or set WANDB_ENTITY."
            )
        return f"{entity}/{run_id}"

    entity = entity or os.getenv("WANDB_ENTITY")
    project = project or os.getenv("WANDB_PROJECT", "momos")
    if entity is None:
        raise ValueError(
            "Run id without slashes requires both entity and project context. "
            "Pass entity/project or set WANDB_ENTITY and WANDB_PROJECT."
        )
    return f"{entity}/{project}/{run_id}"


def _pick_wandb_model_artifact(api, run, artifact_ref=None):
    """Resolve the model artifact for a run."""
    if artifact_ref is not None:
        return api.artifact(str(artifact_ref)), str(artifact_ref)

    # Primary path: this key is written by our training script.
    summary_ref = run.summary.get("wandb_checkpoint_artifact")
    if isinstance(summary_ref, str) and summary_ref.strip():
        return api.artifact(summary_ref), summary_ref

    # Secondary path: reconstruct from run config metadata if present.
    wb_cfg = run.config.get("wandb", {})
    artifact_name = wb_cfg.get("artifact_name")
    if isinstance(artifact_name, str) and artifact_name.strip():
        fallback_ref = f"{run.entity}/{run.project}/{artifact_name}:latest"
        return api.artifact(fallback_ref), fallback_ref

    # Final fallback: choose latest model artifact logged by the run.
    model_artifacts = [
        art for art in list(run.logged_artifacts()) if str(getattr(art, "type", "")) == "model"
    ]
    if not model_artifacts:
        raise ValueError(
            f"No model artifacts found for run {run.path}. "
            "Make sure the run was executed with --wandb."
        )

    def _version_key(artifact):
        version = str(getattr(artifact, "version", ""))
        if version.startswith("v") and version[1:].isdigit():
            return int(version[1:])
        return -1

    artifact = sorted(model_artifacts, key=_version_key)[-1]
    ref = str(getattr(artifact, "qualified_name", "")) or str(getattr(artifact, "name", ""))
    return artifact, ref


def download_wandb_run_artifacts(
    run_id,
    entity=None,
    project=None,
    artifact_ref=None,
    download_root=None,
):
    """Download the model artifact for a W&B run id/path/URL."""
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError("download_wandb_run_artifacts requires wandb package") from exc

    run_path = _resolve_wandb_run_path(run_id, entity=entity, project=project)
    if download_root is None:
        repo_root = Path(__file__).resolve().parents[1]
        download_root = repo_root / "wandb_artifacts"

    api = wandb.Api()
    run = api.run(run_path)
    artifact, resolved_ref = _pick_wandb_model_artifact(api, run, artifact_ref=artifact_ref)
    artifact_dir = Path(artifact.download(root=str(download_root)))

    return {
        "artifact_dir": str(artifact_dir),
        "artifact_ref": str(resolved_ref),
        "run_path": str(run.path),
        "run_id": str(run.id),
        "run_url": str(getattr(run, "url", f"https://wandb.ai/{run.path}")),
    }


def load_wandb_checkpoint_from_run_id(
    run_id,
    checkpoint="best",
    entity=None,
    project=None,
    artifact_ref=None,
    map_location="cpu",
    download_root=None,
):
    """Load one checkpoint payload from W&B directly via run id/path/URL."""
    out = download_wandb_run_artifacts(
        run_id=run_id,
        entity=entity,
        project=project,
        artifact_ref=artifact_ref,
        download_root=download_root,
    )
    artifact_dir = Path(out["artifact_dir"])

    ckpt_name = str(checkpoint)
    if not ckpt_name.endswith(".pt"):
        ckpt_name = f"{ckpt_name}.pt"
    ckpt_path = artifact_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint '{ckpt_name}' not found in artifact {out['artifact_ref']}")

    payload = torch.load(ckpt_path, map_location=map_location)
    out["checkpoint"] = payload
    out["checkpoint_path"] = str(ckpt_path)
    return out


def load_model_from_wandb_run_id(
    model,
    run_id,
    checkpoint="best",
    entity=None,
    project=None,
    artifact_ref=None,
    map_location="cpu",
    download_root=None,
    strict=True,
):
    """Load W&B checkpoint weights into the provided model via run id/path/URL."""
    out = load_wandb_checkpoint_from_run_id(
        run_id=run_id,
        checkpoint=checkpoint,
        entity=entity,
        project=project,
        artifact_ref=artifact_ref,
        map_location=map_location,
        download_root=download_root,
    )
    payload = out["checkpoint"]
    state = payload.get("model", payload)
    load_info = model.load_state_dict(state, strict=strict)
    out["missing_keys"] = list(getattr(load_info, "missing_keys", []))
    out["unexpected_keys"] = list(getattr(load_info, "unexpected_keys", []))
    return out
