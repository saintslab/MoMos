import inspect
import torch.nn as nn
import resnet_s
import timm


class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


def build_local_resnet(name, num_classes):
    if not str(name).startswith("resnet"):
        return None
    fn = getattr(resnet_s, name, None)
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    if "num_classes" in sig.parameters:
        return fn(num_classes=num_classes)
    return fn()


def get_model(name, num_classes):
    if name == "mlp":
        return MLP(3 * 32 * 32, num_classes)

    local_resnet = build_local_resnet(name, num_classes)
    if local_resnet is not None:
        return local_resnet

    if timm is None:
        raise ValueError(f"Unknown model {name} and timm is not installed")
    return timm.create_model(name, num_classes=num_classes, pretrained=False)
