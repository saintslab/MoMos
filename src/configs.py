dataset_defaults = {
    "cifar10": {"num_classes": 10, "in_channels": 3, "img_size": 32, "val_pct": 0.05, "test_pct": 1.0},
    "mnist": {"num_classes": 10, "in_channels": 1, "img_size": 28, "val_pct": None, "test_pct": 1.0},
    "fashion_mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "img_size": 28,
        "val_pct": None,
        "test_pct": 1.0,
    },
}

base_profile = {
    "epochs": 200,
    "patience": None,
    "batch_size": 128,
    "momentum": 0.9,
    "lr_scheduler": "cosine",
}

model_profiles = {
    "resnet": {
        **base_profile,
        "optimizer": "sgd",
        "learning_rate": 0.1,
        "weight_decay": 1e-4,
    },
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
    "mnist_resnet": {"dataset": "mnist", "profile": "resnet"},
    "mnist_vit": {"dataset": "mnist", "profile": "vit"},
    "mnist_mlp": {"dataset": "mnist", "profile": "mlp"},
    "fashion_mnist_resnet": {"dataset": "fashion_mnist", "profile": "resnet"},
    "fashion_mnist_vit": {"dataset": "fashion_mnist", "profile": "vit"},
    "fashion_mnist_mlp": {"dataset": "fashion_mnist", "profile": "mlp"},
}


def profile_for_model(model_name):
    """Map a model identifier to a training-profile key.

    Args:
        model_name: Model string from CLI/config.

    Returns:
        Profile key used to select optimizer and LR defaults.
    """
    if model_name == "mlp":
        return "mlp"
    if model_name.startswith("vit"):
        return "vit"
    if model_name.startswith("resnet"):
        return "resnet"
    return "resnet"


def resolve(model, config_name=None):
    """Build a full run config by merging dataset + profile defaults.

    Args:
        model: Model string requested for the run.
        config_name: Optional named preset from ``named_configs``.

    Returns:
        A configuration dictionary consumed by training.
    """
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
