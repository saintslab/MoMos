import argparse
import os
import random
import sys
import time
import torch
import configs
import datasets
import metrics
import models
import quantizers
import utils
from logger import Logger


def build_optimizer(model, cfg, device):
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
    name = str(cfg.get("lr_scheduler", "none")).lower()
    if name in ["none", "off", "null"]:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["epochs"]))
    raise ValueError(f"Unsupported lr scheduler: {name}")


def make_progress(total_steps, max_updates, min_interval):
    return {
        "every": max(1, total_steps // max(1, int(max_updates))),
        "last": 0.0,
        "min_interval": float(min_interval),
        "start": time.perf_counter(),
        "live": sys.stdout.isatty(),
        "width": 0,
    }


def print_progress(label, step, total_steps, state):
    if not state["live"]:
        return
    if step != total_steps and step % state["every"] != 0:
        return

    now = time.perf_counter()
    if step != total_steps and (now - state["last"]) < state["min_interval"]:
        return
    state["last"] = now

    elapsed = now - state["start"]
    eta = (elapsed / max(1, step)) * max(0, total_steps - step)
    pct = 100.0 * step / max(1, total_steps)
    msg = f"{label} {step}/{total_steps} ({pct:5.1f}%) eta {utils.format_seconds(eta)}"
    pad = max(0, state["width"] - len(msg))
    state["width"] = max(state["width"], len(msg))
    print(f"\r{msg}{' ' * pad}", end="", flush=True)


def clear_progress(state):
    if not state["live"] or state["width"] == 0:
        return
    print(f"\r{' ' * state['width']}\r", end="", flush=True)
    state["width"] = 0


def move_batch(data, target, device, transfer):
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
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    transfer = transfer or {"non_blocking": False, "channels_last": False}

    state = make_progress(
        len(loader),
        max_updates=20 if is_train else 10,
        min_interval=0.4 if is_train else 0.5,
    )

    total_loss = 0.0
    correct = 0
    total = 0

    q_epoch_metrics = {}
    motif_counts = None

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

                if quant_cfg is not None:
                    q_metrics = quantizers.quantize(model, quant_cfg)
                    if q_metrics:
                        for key, value in q_metrics.items():
                            if key == "motif_counts":
                                continue
                            if key == "q_time":
                                try:
                                    q_epoch_metrics[key] = q_epoch_metrics.get(key, 0.0) + float(value)
                                except (TypeError, ValueError):
                                    pass
                            else:
                                try:
                                    if key == "num_changed_weights":
                                        q_epoch_metrics[key] = int(value)
                                    else:
                                        q_epoch_metrics[key] = float(value)
                                except (TypeError, ValueError):
                                    pass
                        if "motif_counts" in q_metrics:
                            counts = q_metrics["motif_counts"]
                            if not torch.is_tensor(counts):
                                counts = torch.tensor(counts, dtype=torch.long)
                            counts = counts.to("cpu", dtype=torch.long)
                            motif_counts = counts.clone()

            total_loss += float(loss.item())
            pred = output.argmax(dim=1)
            total += pred.size(0)
            correct += int(pred.eq(target).sum().item())

            print_progress(label, step, len(loader), state)

    clear_progress(state)
    q_epoch = q_epoch_metrics if q_epoch_metrics else None
    if q_epoch is not None and motif_counts is not None:
        q_epoch["motif_counts"] = motif_counts.tolist()
    return total_loss / max(1, len(loader)), correct / max(1, total), q_epoch


def build_quant_config(args, model):
    method = args.method.lower() if args.method is not None else None
    if method in ["none", "baseline"]:
        return None

    uses_momos_args = args.s is not None or args.k is not None or args.capacity is not None or args.force_zero
    wants_quantization = bool(args.quantize) or method is not None
    if not wants_quantization:
        return None

    if method is None:
        method = "momos" if uses_momos_args else "qat"

    if method == "qat":
        return {"method": "qat", "q": int(args.q)}

    if method != "momos":
        raise ValueError(f"Unknown quantization method: {method}. Available: none, {', '.join(quantizers.available_methods())}")

    if args.s is None:
        raise ValueError("--s is required for momos")
    if args.k is None and args.capacity is None:
        raise ValueError("Set --k or --capacity for momos")

    k = int(args.k) if args.k is not None else quantizers.k_from_capacity(model, args.s, args.capacity)
    return {
        "method": "momos",
        "s": int(args.s),
        "k": int(k),
        "q": int(args.q),
        "force_zero": bool(args.force_zero),
        "capacity": args.capacity,
    }


def parse_args():
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
        ("method", str, None),
        ("metrics", str, None),
        ("seed", int, None),
        ("gpu", str, None),
        ("num_workers", int, None),
    ]:
        parser.add_argument(f"--{name}", type=typ, default=default)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--force_zero", action="store_true", default=False)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--all_compression_metrics_binarized", action="store_true", default=False)
    return parser.parse_args()


def main():
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
    runtime = utils.runtime_profile(device, num_workers=args.num_workers, pin_memory=args.pin_memory)
    transfer = runtime.get("transfer", {"non_blocking": False, "channels_last": False})
    cfg["runtime"] = {
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
    }

    if args.metrics is None:
        cfg["metrics"] = list(cfg.get("metrics", []))
    elif args.metrics.strip() == "":
        cfg["metrics"] = []
    else:
        cfg["metrics"] = [item.strip() for item in args.metrics.split(",") if item.strip()]
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

    model = models.get_model(cfg["model"], cfg["num_classes"])
    quant_cfg = build_quant_config(args, model)
    if quant_cfg is not None and quant_cfg.get("method") == "momos":
        quant_cfg["total_num_blocks"] = quantizers.count_total_blocks(model, quant_cfg["s"])
    cfg["quantization"] = quant_cfg

    exp_name = utils.make_experiment_name(cfg["model"], cfg["dataset"], quant_cfg, args.prefix)
    run_dir = utils.next_run_dir(args.logs_dir, exp_name)
    logger = Logger(run_dir)

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
        "val_pct": utils.normalize_pct(cfg.get("val_pct"), "val_pct"),
        "test_pct": utils.normalize_pct(cfg.get("test_pct"), "test_pct"),
        "split_seed": int(args.split_seed),
        "resolved_split": split_info,
    }

    cfg["model_num_params"] = int(sum(param.numel() for param in model.parameters()))

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

    run_start = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch_log = None
    last_epoch_log = None
    patience_counter = 0
    epochs_ran = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
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
        )
        train_time = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            transfer,
            label="val",
        )
        val_time = time.perf_counter() - val_start

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
            epoch_log["metrics"] = metrics.compute_metrics(
                model,
                cfg["metrics"],
                compression_binarized=cfg.get("all_compression_metrics_binarized", False),
            )

        logger.log_epoch(epoch_log)
        last_epoch_log = epoch_log

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_log = epoch_log
            patience_counter = 0
        else:
            patience_counter += 1

        elapsed = time.perf_counter() - run_start
        eta = (elapsed / epoch) * (int(cfg["epochs"]) - epoch)
        print(
            f"{epoch}/{cfg['epochs']} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | eta_run={utils.format_seconds(eta)}"
        )

        patience = cfg.get("patience")
        if patience is not None and int(patience) > 0 and patience_counter >= int(patience):
            print(f"Early stopping at epoch {epoch}")
            break

    logger.save_checkpoint(model, "final.pt", optimizer=optimizer, scheduler=scheduler, epoch=epochs_ran)

    if split_info.get("has_proper_test", False):
        test_loss, test_acc, _ = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            transfer,
            label="test",
        )
        test_evaluated = True
        print(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
    else:
        test_loss, test_acc = None, None
        test_evaluated = False
        print("Final test evaluation skipped: no holdout test split configured.")

    empty_log = {"epoch": 0, "train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None}
    best_epoch_log = best_epoch_log or empty_log
    last_epoch_log = last_epoch_log or empty_log

    logger.log_summary(
        {
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
            "wall_time": time.perf_counter() - run_start,
        }
    )


if __name__ == "__main__":
    main()
