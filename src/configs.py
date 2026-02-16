dataset_defaults = {
    "cifar10": {"num_classes": 10, "img_size": 32, "val_pct": 0.05, "test_pct": 1.0},
    "mnist": {"num_classes": 10, "img_size": 28, "val_pct": None, "test_pct": 1.0},
    "fashion_mnist": {"num_classes": 10, "img_size": 28, "val_pct": None, "test_pct": 1.0},
}

base_profile = {
    "epochs": 200,
    "patience": None,
    "batch_size": 128,
    "momentum": 0.9,
    "lr_scheduler": "cosine",
}

model_profiles = {
    "resnet": {**base_profile, "optimizer": "sgd", "learning_rate": 0.1, "weight_decay": 1e-4},
    "vit": {
        **base_profile,
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "img_size": 224,
    },
    "mlp": {
        **base_profile,
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "img_size": 32,
    },
}

named_configs = {
    "cifar10_resnet": {"dataset": "cifar10", "profile": "resnet"},
    "cifar10_vit": {"dataset": "cifar10", "profile": "vit"},
    "cifar10_mlp": {"dataset": "cifar10", "profile": "mlp"},
}


def profile_for_model(model_name):
    if model_name == "mlp":
        return "mlp"
    if model_name.startswith("vit"):
        return "vit"
    if model_name.startswith("resnet"):
        return "resnet"
    return "resnet"


def resolve(model, config_name=None):
    if config_name is None:
        dataset = "cifar10"
        profile = profile_for_model(model)
    else:
        preset = named_configs.get(config_name)
        if preset is None:
            raise ValueError(f"Unknown config: {config_name}")
        dataset = preset["dataset"]
        profile = preset["profile"]

    if dataset not in dataset_defaults:
        raise ValueError(f"Unsupported dataset: {dataset}")

    cfg = {
        **dataset_defaults[dataset],
        **model_profiles[profile],
        "dataset": dataset,
        "model": model,
        "metrics": [],
    }
    return cfg
