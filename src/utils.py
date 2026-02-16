import json
import random
from pathlib import Path

import torch
import numpy as np


def normalize_pct(value, name):
    if value is None:
        return None
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0,1] or (0,100]. Got {value}")
    return value


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode):
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


def runtime_profile(device, num_workers=None, pin_memory=None):
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

    if cfg["num_workers"] <= 0:
        cfg["num_workers"] = 0
        cfg["persistent_workers"] = False
        cfg["prefetch_factor"] = None
    return cfg


def configure_cuda_fast_path():
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
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def next_run_dir(logs_dir, exp_name):
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
    print(f"Run: exp={exp_name} | dir={run_dir}")
    print(f"Data: model={cfg['model']} | dataset={cfg['dataset']} | device={cfg['device']} | seed={cfg['seed']}")

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

    q = cfg.get("quantization")
    if not q:
        print("Method: baseline")
        return

    method = str(q.get("method", "qat")).lower()

    if method == "qat":
        msg = f"Method: qat | q={q.get('q', 32)}"
        print(msg)
        return

    if method == "momos":
        msg = f"Method: momos | q={q.get('q', 32)} | s={q['s']} | k={q['k']} | force_zero={q.get('force_zero', False)}"
        if q.get("capacity") is not None:
            msg += f" | capacity={q['capacity']}"
        print(msg)
        return

    msg = f"Method: {method}"
    if q.get("q") is not None:
        msg += f" | q={q.get('q')}"
    print(msg)


def load_runs(logs_dir="logs", completed_only=False):
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

            run["_exp_name"] = exp_dir.name
            run["_run_dir"] = str(run_dir)
            run["_results_path"] = str(path)
            runs.append(run)

    return runs


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
        if method is not None and q_method != method:
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
