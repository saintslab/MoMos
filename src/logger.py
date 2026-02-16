import json
from pathlib import Path
import torch


class Logger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "results.json"
        self.motif_path = self.run_dir / "motifs_dist.json"
        self.log = {
            "config": {},
            "epochs": [],
            "summary": {"completed": False},
            "artifacts": {},
        }
        self.motif_log = {"motif_usage": []}

    def set_config(self, cfg):
        self.log["config"] = cfg
        self.save()

    def log_epoch(self, epoch_data):
        self.log["epochs"].append(epoch_data)
        self.save()

    def log_summary(self, summary_data):
        self.log["summary"].update(summary_data)
        self.save()

    def save_checkpoint(self, model, name, optimizer=None, scheduler=None, epoch=None):
        state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        payload = {"model": state_model.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
            payload["lr"] = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            payload["epoch"] = int(epoch)
        if self.log.get("config"):
            payload["config"] = self.log["config"]
        torch.save(payload, self.run_dir / name)
        self.log["artifacts"][Path(name).stem] = name
        self.save()

    def log_motif_usage(self, data):
        self.motif_log["motif_usage"].append(
            {
                "epoch": data.get("epoch"),
                "motif_counts": data.get("motif_counts"),
            }
        )
        with open(self.motif_path, "w") as f:
            json.dump(self.motif_log, f, indent=2)
        self.log["artifacts"]["motifs_dist"] = "motifs_dist.json"
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.log, f, indent=2)
