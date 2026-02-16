import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import utils


def build_transform(dataset_name, img_size, is_train):
    if dataset_name in ["mnist", "fashion_mnist"]:
        ops = []
        if img_size != 28:
            ops.append(transforms.Resize((img_size, img_size)))
        ops.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return transforms.Compose(ops)

    if dataset_name == "cifar10":
        ops = []
        if is_train:
            ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
        if img_size != 32:
            ops.append(transforms.Resize((img_size, img_size)))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        return transforms.Compose(ops)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_dataset(dataset_name, is_train, transform, data_dir):
    if dataset_name == "cifar10":
        return datasets.CIFAR10(data_dir, train=is_train, transform=transform, download=True)
    if dataset_name == "mnist":
        return datasets.MNIST(data_dir, train=is_train, transform=transform, download=True)
    if dataset_name == "fashion_mnist":
        return datasets.FashionMNIST(data_dir, train=is_train, transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_loader(dataset_obj, batch_size, shuffle, runtime):
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(runtime["persistent_workers"])
        if runtime["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = int(runtime["prefetch_factor"])
    return DataLoader(dataset_obj, **kwargs)


def count_from_pct(total, pct, name):
    pct = utils.normalize_pct(pct, name)
    if pct is None:
        return None
    count = int(total * pct)
    if count <= 0:
        raise ValueError(f"{name} is too small and yields zero samples")
    return count


def get_dataloaders(
    dataset_name,
    batch_size,
    img_size,
    val_pct=None,
    test_pct=None,
    split_seed=10,
    runtime=None,
    data_dir="./data",
):
    runtime = runtime or {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }

    train_data = load_dataset(dataset_name, True, build_transform(dataset_name, img_size, True), data_dir)
    train_eval_data = load_dataset(dataset_name, True, build_transform(dataset_name, img_size, False), data_dir)
    test_data = load_dataset(dataset_name, False, build_transform(dataset_name, img_size, False), data_dir)

    total_train = len(train_data)
    total_test = len(test_data)

    rng = torch.Generator().manual_seed(int(split_seed))
    perm_train = torch.randperm(total_train, generator=rng)
    perm_test = torch.randperm(total_test, generator=rng)

    val_count = count_from_pct(total_train, val_pct, "val_pct")

    if val_count is None:
        train_data_final = train_data
        eval_count = count_from_pct(total_test, test_pct if test_pct is not None else 1.0, "test_pct")
        eval_idx = perm_test[:eval_count].tolist()
        val_data = Subset(test_data, eval_idx)
        test_data_final = Subset(test_data, eval_idx)
        split_mode = "train_plus_test_as_validation"
        has_proper_test = False
        resolved_train = total_train
        resolved_val = eval_count
        resolved_test = eval_count
    else:
        train_count = total_train - val_count
        if train_count <= 0:
            raise ValueError("val_pct leaves no training samples")
        val_idx = perm_train[:val_count].tolist()
        train_idx = perm_train[val_count:].tolist()
        train_data_final = Subset(train_data, train_idx)
        val_data = Subset(train_eval_data, val_idx)
        test_count = count_from_pct(total_test, test_pct if test_pct is not None else 1.0, "test_pct")
        test_idx = perm_test[:test_count].tolist()
        test_data_final = Subset(test_data, test_idx)
        split_mode = "train_val_from_train_test"
        has_proper_test = True
        resolved_train = train_count
        resolved_val = val_count
        resolved_test = test_count

    split_info = {
        "split_mode": split_mode,
        "has_proper_test": has_proper_test,
        "train_size": resolved_train,
        "val_size": resolved_val,
        "test_size": resolved_test,
        "val_pct": utils.normalize_pct(val_pct, "val_pct"),
        "test_pct": utils.normalize_pct(test_pct, "test_pct"),
        "split_seed": int(split_seed),
    }

    train_loader = build_loader(train_data_final, batch_size, True, runtime)
    val_loader = build_loader(val_data, batch_size, False, runtime)
    test_loader = build_loader(test_data_final, batch_size, False, runtime)
    return train_loader, val_loader, test_loader, split_info
