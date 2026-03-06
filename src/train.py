import argparse
import os
import random
import time
import torch
import configs
import datasets
import metrics
import models
import quantizers
import utils
from logger import Logger


def _init_wandb_if_enabled(
    enabled,
    cfg,
    quant_cfg,
    exp_name,
    run_dir,
    wandb_project=None,
    wandb_entity=None,
):
    """Initialize W&B for this run and return module, run handle, and metadata."""
    if not enabled:
        return None, None, {"enabled": False}

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError("--wandb requires the wandb package to be installed") from exc

    method_label, _ = utils.quantization_overview(quant_cfg)
    tags = [
        f"model:{cfg['model']}",
        f"dataset:{cfg['dataset']}",
        f"method:{method_label}",
    ]
    if quant_cfg is not None:
        if quant_cfg.get("q") is not None:
            tags.append(f"q:{quant_cfg.get('q')}")
        if quant_cfg.get("s") is not None:
            tags.append(f"s:{quant_cfg.get('s')}")
        if quant_cfg.get("capacity") is not None:
            tags.append(f"capacity:{quant_cfg.get('capacity')}")
        if quant_cfg.get("force_zero") is not None:
            tags.append(f"force_zero:{int(bool(quant_cfg.get('force_zero')))}")

    project = str(wandb_project) if wandb_project is not None else os.getenv("WANDB_PROJECT", "momos")
    entity = str(wandb_entity) if wandb_entity is not None else os.getenv("WANDB_ENTITY", None)
    wandb_dir = os.getenv("WANDB_DIR")
    if not wandb_dir:
        # Default to repo-local folder so it is easy to inspect and delete.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        wandb_dir = os.path.join(repo_root, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    run_name = f"{cfg['model']}-{method_label}-{run_dir.name}"
    wb_run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=exp_name,
        tags=tags,
        config=cfg,
        save_code=False,
        job_type="train",
        dir=wandb_dir,
    )

    artifact_name = f"{exp_name}-{run_dir.name}-checkpoints"
    wb_meta = {
        "enabled": True,
        "run_id": wb_run.id,
        "run_name": wb_run.name,
        "run_path": wb_run.path,
        "url": wb_run.url,
        "project": wb_run.project,
        "entity": wb_run.entity,
        "group": exp_name,
        "tags": tags,
        "artifact_name": artifact_name,
        "dir": wandb_dir,
    }
    return wandb, wb_run, wb_meta


def _wandb_epoch_payload(epoch_log):
    """Flatten one epoch log into scalar W&B keys."""
    out = {}

    def _visit(prefix, value):
        if isinstance(value, dict):
            for key, sub in value.items():
                child = f"{prefix}/{key}" if prefix else str(key)
                _visit(child, sub)
            return
        if isinstance(value, bool):
            out[prefix] = int(value)
            return
        if isinstance(value, (int, float)):
            out[prefix] = float(value)

    _visit("", epoch_log)
    return out


def _wandb_summary_payload(summary_data):
    """Filter summary dict to scalar values for W&B summary."""
    out = {}
    for key, value in summary_data.items():
        if isinstance(value, bool):
            out[key] = bool(value)
        elif isinstance(value, (int, float, str)) or value is None:
            out[key] = value
    return out


def _wandb_upload_artifacts(wandb, wb_run, run_dir, artifact_name, cfg):
    """Upload run artifacts once at end of training."""
    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={
            "model": cfg.get("model"),
            "dataset": cfg.get("dataset"),
            "method": utils.quantization_overview(cfg.get("quantization"))[0],
            "seed": cfg.get("seed"),
        },
    )

    # Required checkpoints.
    for name in ["init.pt", "best.pt", "final.pt", "results.json"]:
        path = run_dir / name
        if path.exists():
            artifact.add_file(str(path), name=name)

    # Optional MoMos artifacts.
    for name in ["motifs_dist.json", "momos_state.pt"]:
        path = run_dir / name
        if path.exists():
            artifact.add_file(str(path), name=name)

    wb_run.log_artifact(artifact, aliases=["latest", run_dir.name])
    return f"{wb_run.entity}/{wb_run.project}/{artifact_name}:latest"


def build_optimizer(model, cfg, device):
    """Build optimizer from config, with CUDA fused/foreach fallbacks."""
    name = str(cfg["optimizer"]).lower()
    lr = float(cfg["learning_rate"])
    wd = float(cfg["weight_decay"])

    if name == "sgd":
        opt_cls = torch.optim.SGD
        kwargs = {"lr": lr, "weight_decay": wd, "momentum": float(cfg.get("momentum", 0.9))}
    elif name == "adam":
        opt_cls = torch.optim.Adam
        kwargs = {"lr": lr, "weight_decay": wd}
    elif name == "adamw":
        opt_cls = torch.optim.AdamW
        kwargs = {"lr": lr, "weight_decay": wd}
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    if device != "cuda":
        return opt_cls(model.parameters(), **kwargs)

    for extra in ({"fused": True}, {"foreach": True}):
        try:
            return opt_cls(model.parameters(), **extra, **kwargs)
        except TypeError:
            continue
    return opt_cls(model.parameters(), **kwargs)


def build_scheduler(optimizer, cfg):
    """Build LR scheduler from config (or return ``None``)."""
    name = str(cfg.get("lr_scheduler", "none")).lower()
    if name in ["none", "off", "null"]:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    raise ValueError(f"Unsupported lr scheduler: {name}")


def maybe_log_step(progress_mode, step, total_steps, step_every, label, epoch_idx, epochs_total, avg_loss, avg_acc):
    """Print coarse train-step progress when ``--progress step`` is active."""
    if progress_mode != "step":
        return
    if label != "train":
        return
    if total_steps <= 0 or step_every <= 0:
        return
    if step != total_steps and step % step_every != 0:
        return
    print(
        f"{label} step {step}/{total_steps} "
        f"loss={avg_loss:.4f} acc={avg_acc:.4f}"
    )


def print_epoch_summary(
    epoch,
    total_epochs,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    lr,
    train_time,
    val_time,
    best_val_loss,
    best_val_acc,
    best_epoch,
    improved,
    patience_counter,
    patience,
):
    """Print one compact epoch-level status line."""
    def _fmt_metric(value, fmt_spec):
        if value is None:
            return "-"
        return format(float(value), fmt_spec)

    def _fmt_time(value):
        if value is None:
            return "-"
        return format(float(value), ".1f")

    improved_text = "yes" if improved else "no"
    best_epoch_text = "-" if best_epoch is None else str(int(best_epoch))
    best_val_acc_text = "-" if best_val_acc is None else f"{float(best_val_acc):.4f}"
    msg = (
        f"[epoch {epoch:>3}/{int(total_epochs):>3}] "
        f"train loss={_fmt_metric(train_loss, '.4f')} acc={_fmt_metric(train_acc, '.4f')} | "
        f"val loss={_fmt_metric(val_loss, '.4f')} acc={_fmt_metric(val_acc, '.4f')} | "
        f"lr={lr:.3e} | "
        f"time train/val={_fmt_time(train_time)}/{_fmt_time(val_time)}s | "
        f"improved={improved_text} | "
        f"best_epoch={best_epoch_text} best_val_loss={best_val_loss:.4f} best_val_acc={best_val_acc_text}"
    )
    if patience is not None and int(patience) > 0:
        msg += f" | patience={int(patience_counter)}/{int(patience)}"
    print(msg)


def move_batch(data, target, device, transfer):
    """Move one `(data, target)` batch to device using runtime transfer settings."""
    non_blocking = transfer.get("non_blocking", False)
    if transfer.get("channels_last", False) and data.dim() == 4:
        data = data.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
    else:
        data = data.to(device, non_blocking=non_blocking)
    target = target.to(device, non_blocking=non_blocking)
    return data, target


def run_epoch(
    model,
    loader,
    criterion,
    device,
    transfer,
    label,
    optimizer=None,
    quant_cfg=None,
    epoch_idx=None,
    epochs_total=None,
    progress_mode="epoch",
    step_updates=5,
):
    """Run one train or eval pass and return `(loss, acc, quant_stats)`."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    transfer = transfer or {"non_blocking": False, "channels_last": False}

    total_steps = len(loader)
    step_every = max(1, total_steps // max(1, int(step_updates))) if total_steps > 0 else 1

    total_loss = 0.0
    correct = 0
    total = 0

    q_epoch_sums = {}
    q_epoch_last = {}
    motif_counts = None
    quant_method = None if quant_cfg is None else str(quant_cfg.get("method", "")).lower()
    defer_momos_to_epoch_end = bool(is_train and quant_method == "momos")
    apply_step_quant = bool(is_train and quant_cfg is not None and quant_method not in {"momos", "qat"})

    def _accumulate_q_metrics(q_metrics):
        nonlocal motif_counts
        for key, value in q_metrics.items():
            if key == "motif_counts":
                continue
            if key == "method":
                q_epoch_last[key] = value
                continue
            if isinstance(value, bool):
                q_epoch_last[key] = bool(value)
                continue
            if isinstance(value, (int, float)):
                try:
                    if key == "num_changed_weights":
                        q_epoch_sums[key] = int(q_epoch_sums.get(key, 0)) + int(value)
                    else:
                        q_epoch_sums[key] = float(q_epoch_sums.get(key, 0.0)) + float(value)
                except (TypeError, ValueError):
                    pass
            else:
                q_epoch_last[key] = value
        if "motif_counts" in q_metrics:
            counts = q_metrics["motif_counts"]
            if not torch.is_tensor(counts):
                counts = torch.tensor(counts, dtype=torch.long)
            counts = counts.to("cpu", dtype=torch.long)
            if motif_counts is None:
                motif_counts = counts.clone()
            else:
                if counts.numel() > motif_counts.numel():
                    motif_counts = torch.cat(
                        [
                            motif_counts,
                            torch.zeros(counts.numel() - motif_counts.numel(), dtype=torch.long),
                        ]
                    )
                elif counts.numel() < motif_counts.numel():
                    counts = torch.cat(
                        [
                            counts,
                            torch.zeros(motif_counts.numel() - counts.numel(), dtype=torch.long),
                        ]
                    )
                motif_counts += counts

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for step, (data, target) in enumerate(loader, start=1):
            data, target = move_batch(data, target, device, transfer)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            output = model(data)
            loss = criterion(output, target)

            if is_train:
                loss.backward()
                optimizer.step()

                # Non-MoMos/non-QAT quantizers may run per optimizer step.
                if apply_step_quant:
                    q_metrics = quantizers.quantize(model, quant_cfg)
                    if q_metrics:
                        _accumulate_q_metrics(q_metrics)

            total_loss += float(loss.item())
            pred = output.argmax(dim=1)
            total += pred.size(0)
            correct += int(pred.eq(target).sum().item())

            maybe_log_step(
                progress_mode=progress_mode,
                step=step,
                total_steps=total_steps,
                step_every=step_every,
                label=label,
                epoch_idx=epoch_idx,
                epochs_total=epochs_total,
                avg_loss=total_loss / step,
                avg_acc=correct / max(1, total),
            )

    if is_train and quant_cfg is not None and defer_momos_to_epoch_end:
        quant_epoch_cfg = dict(quant_cfg)
        if quant_epoch_cfg.get("chunk_progress", False):
            quant_epoch_cfg["progress_prefix"] = "computing nearest motifs"

        q_metrics = quantizers.quantize(model, quant_epoch_cfg)
        if q_metrics:
            _accumulate_q_metrics(q_metrics)

    q_epoch = None
    if q_epoch_sums or q_epoch_last or motif_counts is not None:
        q_epoch = {}
        q_epoch.update(q_epoch_sums)
        q_epoch.update(q_epoch_last)
        if motif_counts is not None:
            q_epoch["motif_counts"] = motif_counts.tolist()
    return total_loss / max(1, len(loader)), correct / max(1, total), q_epoch


def build_quant_config(args, model):
    """Validate quantization CLI flags and build normalized config dict."""
    method = args.method.lower() if args.method is not None else None
    q_bits = int(args.q)
    if q_bits < 2:
        raise ValueError(f"--q must be >= 2, got {q_bits}")
    chunk_size = args.chunk_size
    uses_momos_args = (
        args.s is not None
        or args.k is not None
        or args.capacity is not None
        or args.force_zero
        or chunk_size is not None
        or args.chunk_progress
        or args.chunk_progress_elements is not None
    )
    uses_q_override = int(args.q) != 32

    if method in ["none", "baseline"]:
        if uses_momos_args or uses_q_override:
            raise ValueError("method=baseline/none conflicts with quantization arguments")
        return None

    wants_quantization = method is not None or uses_momos_args or uses_q_override
    if not wants_quantization:
        return None

    if method is None:
        method = "momos" if uses_momos_args else "qat"

    if method == "qat":
        if uses_momos_args:
            raise ValueError(
                "method=qat conflicts with MoMos-only arguments "
                "(--s/--k/--capacity/--force_zero/--chunk_size/--chunk_progress/--chunk_progress_elements)"
            )
        return {"method": "qat", "q": q_bits}

    if method != "momos":
        raise ValueError(f"Unknown quantization method: {method}. Available: none, {', '.join(quantizers.available_methods())}")

    if args.s is None:
        raise ValueError("--s is required for momos")
    if int(args.s) <= 0:
        raise ValueError(f"--s must be > 0, got {args.s}")
    if args.k is None and args.capacity is None:
        raise ValueError("Set --k or --capacity for momos")
    if args.k is not None and int(args.k) <= 0:
        raise ValueError(f"--k must be > 0, got {args.k}")
    if chunk_size is not None and float(chunk_size) <= 0:
        raise ValueError("--chunk_size must be > 0")
    if args.chunk_progress_elements is not None and int(args.chunk_progress_elements) <= 0:
        raise ValueError("--chunk_progress_elements must be > 0")

    k = int(args.k) if args.k is not None else quantizers.k_from_capacity(model, args.s, args.capacity)
    return {
        "method": "momos",
        "s": int(args.s),
        "k": int(k),
        "q": q_bits,
        "force_zero": bool(args.force_zero),
        "capacity": args.capacity,
        "chunk_size": (float(chunk_size) if chunk_size is not None else None),
        "chunk_progress": bool(args.chunk_progress),
        "chunk_progress_elements": (
            int(args.chunk_progress_elements) if args.chunk_progress_elements is not None else None
        ),
    }


def default_qat_exclude_layers(model_name):
    """Return model-family QAT exclusion tokens (name-based)."""
    name = str(model_name).lower()
    if name.startswith("resnet"):
        return ["bn"]
    if name.startswith("vit"):
        return ["norm"]
    return []


def parse_args():
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Minimal training script")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    for name, typ, default in [
        ("data_dir", str, "./data"),
        ("logs_dir", str, "./logs"),
        ("prefix", str, ""),
        ("epochs", int, None),
        ("patience", int, None),
        ("val_pct", float, None),
        ("test_pct", float, None),
        ("split_seed", int, 10),
        ("q", int, 32),
        ("s", int, None),
        ("k", int, None),
        ("capacity", float, None),
        ("chunk_size", float, None),
        ("chunk_progress_elements", int, None),
        ("method", str, None),
        ("metrics", str, None),
        ("seed", int, None),
        ("gpu", str, None),
        ("num_workers", int, None),
        ("prefetch_factor", int, None),
        ("step_updates", int, 5),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=default)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--force_zero", action="store_true", default=False)
    parser.add_argument("--chunk_progress", action="store_true", default=False)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable lightweight epoch-level Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name for this run (overrides WANDB_PROJECT).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team for this run (overrides WANDB_ENTITY).",
    )
    parser.add_argument("--all_compression_metrics_binarized", action="store_true", default=False)
    parser.add_argument(
        "--progress",
        type=str,
        default="epoch",
        choices=["none", "epoch", "step"],
        help="Controls step-level progress logging; epoch summary is always printed.",
    )
    return parser.parse_args()


def main():
    """Run training, validation, optional testing, and logging."""
    main_start = time.perf_counter()
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = configs.resolve(args.model, args.config)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.patience is not None:
        cfg["patience"] = int(args.patience)
    if args.val_pct is not None:
        cfg["val_pct"] = args.val_pct
    if args.test_pct is not None:
        cfg["test_pct"] = args.test_pct

    seed = int(args.seed) if args.seed is not None else random.randint(0, 2**32 - 1)
    cfg["seed"] = seed
    utils.seed_all(seed)

    device = utils.resolve_device(args.device)
    cfg["device"] = device
    runtime = utils.runtime_profile(
        device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    transfer = runtime.get("transfer", {"non_blocking": False, "channels_last": False})
    cfg["runtime"] = {
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
        "persistent_workers": bool(runtime["persistent_workers"]),
        "prefetch_factor": runtime["prefetch_factor"],
    }
    cfg["progress"] = {
        "mode": args.progress,
        "step_updates": int(args.step_updates),
    }

    if args.metrics is None:
        cfg["metrics"] = list(cfg.get("metrics", []))
    elif args.metrics.strip() == "":
        cfg["metrics"] = []
    else:
        cfg["metrics"] = [item.strip() for item in args.metrics.split(",") if item.strip()]
    unknown_metrics = sorted(set(name for name in cfg["metrics"] if name not in metrics.registry))
    if unknown_metrics:
        known = ", ".join(sorted(metrics.registry.keys()))
        bad = ", ".join(unknown_metrics)
        raise ValueError(f"Unknown metric(s): {bad}. Available: {known}")
    cfg["all_compression_metrics_binarized"] = bool(args.all_compression_metrics_binarized)
    if cfg["all_compression_metrics_binarized"]:
        for name in ["bdm", "gzip", "bz2", "lzma"]:
            if name not in cfg["metrics"]:
                cfg["metrics"].append(name)

    cfg["compile"] = bool(args.compile)
    if cfg["compile"] and device != "cuda":
        raise RuntimeError("--compile requires CUDA device")

    if device == "cuda":
        utils.configure_cuda_fast_path()
        cfg["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())

    model = models.get_model(
        cfg["model"],
        cfg["num_classes"],
        img_size=cfg["img_size"],
        in_channels=cfg.get("in_channels", 3),
    )
    quant_cfg = build_quant_config(args, model)
    if quant_cfg is not None:
        q_method = str(quant_cfg.get("method", "")).lower()
        q_bits = int(quant_cfg.get("q", 32))
        use_qat = q_method == "qat" or (q_method == "momos" and q_bits < 32)

        if use_qat:
            quant_cfg.setdefault("exclude_layers", default_qat_exclude_layers(cfg["model"]))
            cfg["qat_setup"] = quantizers.quantize_qat(model, quant_cfg)
            if q_bits < 32:
                attached = int(cfg["qat_setup"].get("attached_modules", 0))
                updated = int(cfg["qat_setup"].get("updated_modules", 0))
                if attached + updated <= 0:
                    raise RuntimeError(
                        "QAT requested but no weight modules were parametrized. "
                        "Check model layers and QAT exclusion rules."
                    )
    if quant_cfg is not None and quant_cfg.get("method") == "momos":
        quant_cfg["total_num_blocks"] = quantizers.count_total_blocks(model, quant_cfg["s"])
    cfg["quantization"] = quant_cfg

    exp_name = utils.make_experiment_name(cfg["model"], cfg["dataset"], quant_cfg, args.prefix)
    run_dir = utils.next_run_dir(args.logs_dir, exp_name)
    logger = Logger(run_dir)

    wandb_module, wb_run, wb_meta = _init_wandb_if_enabled(
        enabled=bool(args.wandb),
        cfg=cfg,
        quant_cfg=quant_cfg,
        exp_name=exp_name,
        run_dir=run_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    cfg["wandb"] = wb_meta

    train_loader, val_loader, test_loader, split_info = datasets.get_dataloaders(
        dataset_name=cfg["dataset"],
        batch_size=cfg["batch_size"],
        img_size=cfg["img_size"],
        val_pct=cfg.get("val_pct"),
        test_pct=cfg.get("test_pct"),
        split_seed=args.split_seed,
        runtime=runtime,
        data_dir=args.data_dir,
    )

    cfg["data"] = {
        "data_dir": args.data_dir,
        "img_size": cfg["img_size"],
        "in_channels": int(cfg.get("in_channels", 3)),
        "val_pct": utils.normalize_pct(cfg.get("val_pct"), "val_pct"),
        "test_pct": utils.normalize_pct(cfg.get("test_pct"), "test_pct"),
        "split_seed": int(args.split_seed),
        "resolved_split": split_info,
    }

    cfg["model_num_params"] = int(sum(param.numel() for param in model.parameters()))
    if wb_run is not None:
        wb_run.config.update(cfg, allow_val_change=True)

    utils.print_run_header(cfg, split_info, exp_name, run_dir)

    logger.set_config(cfg)

    model = model.to(device)
    if transfer.get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    if cfg["compile"]:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            model = torch.compile(model)
        except Exception as exc:
            raise RuntimeError("torch.compile failed") from exc

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg, device)
    scheduler = build_scheduler(optimizer, cfg)
    logger.save_checkpoint(model, "init.pt", optimizer=optimizer, scheduler=scheduler, epoch=0)

    epochs_total = int(cfg["epochs"])
    if quant_cfg is not None and str(quant_cfg.get("method", "")).lower() == "momos" and epochs_total < 1:
        raise ValueError("MoMos runs require --epochs >= 1 to save projected checkpoints.")
    train_time_total = 0.0
    validation_time_total = 0.0

    # Epoch 0: evaluate random initialization before any optimization step.
    # No train-loader pass here; this is validation-only.
    epoch0_val_start = time.perf_counter()
    epoch0_val_loss, epoch0_val_acc, _ = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        transfer,
        label="val",
        epoch_idx=0,
        epochs_total=epochs_total,
        progress_mode="none",
        step_updates=args.step_updates,
    )
    epoch0_val_time = time.perf_counter() - epoch0_val_start
    validation_time_total += epoch0_val_time

    epoch0_log = {
        "epoch": 0,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "train_loss": None,
        "train_acc": None,
        "val_loss": epoch0_val_loss,
        "val_acc": epoch0_val_acc,
        "train_time": None,
        "val_time": epoch0_val_time,
        "train_eval_time": None,
        "val_eval_time": epoch0_val_time,
    }
    if cfg["metrics"]:
        metrics_start = time.perf_counter()
        epoch0_metrics = metrics.compute_metrics(
            model,
            cfg["metrics"],
            compression_binarized=cfg.get("all_compression_metrics_binarized", False),
        )
        epoch0_metrics["time_to_compute"] = time.perf_counter() - metrics_start
        epoch0_log["metrics"] = epoch0_metrics
    logger.log_epoch(epoch0_log)
    if wb_run is not None:
        wb_payload = _wandb_epoch_payload(epoch0_log)
        if wb_payload:
            wb_run.log(wb_payload, step=0)

    best_val_loss = epoch0_val_loss
    best_epoch_log = epoch0_log
    last_epoch_log = epoch0_log
    patience_counter = 0
    epochs_ran = 0

    # For MoMos runs, ensure best.pt is always a post-projection checkpoint.
    # Epoch 0 is validation-only (no MoMos projection), so we defer best.pt
    # saving until the first trained epoch.
    is_momos_run = bool(quant_cfg is not None and str(quant_cfg.get("method", "")).lower() == "momos")
    best_checkpoint_saved = not is_momos_run
    if best_checkpoint_saved:
        logger.save_checkpoint(model, "best.pt", optimizer=optimizer, scheduler=scheduler, epoch=0)

    print_epoch_summary(
        epoch=0,
        total_epochs=epochs_total,
        train_loss=None,
        train_acc=None,
        val_loss=epoch0_val_loss,
        val_acc=epoch0_val_acc,
        lr=float(optimizer.param_groups[0]["lr"]),
        train_time=None,
        val_time=epoch0_val_time,
        best_val_loss=best_val_loss,
        best_val_acc=best_epoch_log.get("val_acc"),
        best_epoch=best_epoch_log.get("epoch"),
        improved=True,
        patience_counter=0,
        patience=cfg.get("patience"),
    )

    for epoch in range(1, epochs_total + 1):
        epochs_ran = epoch

        train_start = time.perf_counter()
        train_loss, train_acc, q_epoch = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            transfer,
            label="train",
            optimizer=optimizer,
            quant_cfg=quant_cfg,
            epoch_idx=epoch,
            epochs_total=epochs_total,
            progress_mode=args.progress,
            step_updates=args.step_updates,
        )
        train_time = time.perf_counter() - train_start
        train_time_total += train_time

        val_start = time.perf_counter()
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            transfer,
            label="val",
            epoch_idx=epoch,
            epochs_total=epochs_total,
            progress_mode="none",
            step_updates=args.step_updates,
        )
        val_time = time.perf_counter() - val_start
        validation_time_total += val_time

        lr = float(optimizer.param_groups[0]["lr"])

        epoch_log = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_time": train_time,
            "val_time": val_time,
        }
        if q_epoch is not None:
            q_epoch_log = dict(q_epoch)
            motif_counts = q_epoch_log.pop("motif_counts", None)
            if motif_counts is not None:
                logger.log_motif_usage(
                    {
                        "epoch": epoch,
                        "motif_counts": motif_counts,
                    }
                )
            epoch_log["quantization"] = q_epoch_log
        if cfg["metrics"]:
            metrics_start = time.perf_counter()
            epoch_metrics = metrics.compute_metrics(
                model,
                cfg["metrics"],
                compression_binarized=cfg.get("all_compression_metrics_binarized", False),
            )
            epoch_metrics["time_to_compute"] = time.perf_counter() - metrics_start
            epoch_log["metrics"] = epoch_metrics

        logger.log_epoch(epoch_log)
        if wb_run is not None:
            wb_payload = _wandb_epoch_payload(epoch_log)
            if wb_payload:
                wb_run.log(wb_payload, step=int(epoch))
        last_epoch_log = epoch_log

        if scheduler is not None:
            scheduler.step()

        if not best_checkpoint_saved:
            improved = True
            best_val_loss = val_loss
            best_epoch_log = epoch_log
            patience_counter = 0
            logger.save_checkpoint(model, "best.pt", optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            best_checkpoint_saved = True
        else:
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch_log = epoch_log
                patience_counter = 0
                logger.save_checkpoint(model, "best.pt", optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            else:
                patience_counter += 1

        patience = cfg.get("patience")
        best_epoch = best_epoch_log.get("epoch") if best_epoch_log is not None else None
        best_val_acc = best_epoch_log.get("val_acc") if best_epoch_log is not None else None
        print_epoch_summary(
            epoch=epoch,
            total_epochs=epochs_total,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=lr,
            train_time=train_time,
            val_time=val_time,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            improved=improved,
            patience_counter=patience_counter,
            patience=patience,
        )

        if patience is not None and int(patience) > 0 and patience_counter >= int(patience):
            print(f"Early stopping at epoch {epoch} (patience {patience_counter}/{int(patience)})")
            break

    logger.save_checkpoint(model, "final.pt", optimizer=optimizer, scheduler=scheduler, epoch=epochs_ran)

    if split_info.get("has_proper_test", False):
        test_start = time.perf_counter()
        test_loss, test_acc, _ = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            transfer,
            label="test",
            epoch_idx=epochs_ran,
            epochs_total=epochs_total,
            progress_mode="none",
            step_updates=args.step_updates,
        )
        test_time = time.perf_counter() - test_start
        test_evaluated = True
        print(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
    else:
        test_loss, test_acc = None, None
        test_time = None
        test_evaluated = False
        print("Final test evaluation skipped: no holdout test split configured.")

    empty_log = {"epoch": 0, "train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None}
    best_epoch_log = best_epoch_log or empty_log
    last_epoch_log = last_epoch_log or empty_log

    wall_time_total = time.perf_counter() - main_start
    training_time_total = train_time_total + validation_time_total
    summary_data = {
        "completed": True,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch_log["epoch"],
        "best_train_loss": best_epoch_log.get("train_loss"),
        "best_train_acc": best_epoch_log.get("train_acc"),
        "best_val_loss": best_epoch_log.get("val_loss"),
        "best_val_acc": best_epoch_log.get("val_acc"),
        "final_epoch": last_epoch_log["epoch"],
        "final_train_loss": last_epoch_log.get("train_loss"),
        "final_train_acc": last_epoch_log.get("train_acc"),
        "final_val_loss": last_epoch_log.get("val_loss"),
        "final_val_acc": last_epoch_log.get("val_acc"),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_evaluated": test_evaluated,
        "test_time": test_time,
        # Full end-to-end time for the whole run, including setup/load overhead.
        "wall_time": wall_time_total,
        # Train + validation time (includes epoch-0 validation pass).
        "training_time": training_time_total,
    }

    if wb_run is not None:
        try:
            artifact_ref = _wandb_upload_artifacts(
                wandb=wandb_module,
                wb_run=wb_run,
                run_dir=run_dir,
                artifact_name=cfg["wandb"]["artifact_name"],
                cfg=cfg,
            )
            summary_data["wandb_checkpoint_artifact"] = artifact_ref
            summary_data["wandb_run_path"] = wb_run.path
            summary_data["wandb_url"] = wb_run.url
        except Exception:
            # Keep training results intact even if artifact upload fails.
            pass

    logger.log_summary(summary_data)

    if wb_run is not None:
        for key, value in _wandb_summary_payload(summary_data).items():
            wb_run.summary[key] = value
        wb_run.finish()
    print(
        "Run complete | "
        f"best_epoch={best_epoch_log['epoch']} "
        f"best_val_acc={best_epoch_log.get('val_acc')} | "
        f"final_val_acc={last_epoch_log.get('val_acc')} | "
        f"test_acc={test_acc} | "
        f"wall_total={utils.format_seconds(wall_time_total)} | "
        f"training={utils.format_seconds(training_time_total)}"
    )


if __name__ == "__main__":
    main()
