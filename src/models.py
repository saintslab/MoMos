import inspect
import torch.nn as nn
import resnet_s
try:
    import timm
except Exception:
    timm = None


class MLP(nn.Module):
    """Simple feed-forward classifier over flattened image pixels."""

    def __init__(self, input_dim, num_classes):
        """Create an MLP classifier for flattened inputs.

        Args:
            input_dim: Flattened input dimension.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        """Run a forward pass after flattening each sample.

        Args:
            x: Input tensor ``(batch, channels, height, width)``.

        Returns:
            Logits tensor ``(batch, num_classes)``.
        """
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x


def build_local_resnet(name, num_classes, in_channels=3):
    """Build a local ResNet implementation when ``name`` matches.

    Args:
        name: Requested model name.
        num_classes: Number of output classes.
        in_channels: Number of input image channels.

    Returns:
        Local model instance when available, else ``None``.
    """
    if not str(name).startswith("resnet"):
        return None
    fn = getattr(resnet_s, name, None)
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn()
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = int(in_channels)
    return fn(**kwargs)


def get_model(name, num_classes, img_size=32, in_channels=3):
    """Create a model instance from local definitions or timm.

    Args:
        name: Model identifier.
        num_classes: Number of output classes.
        img_size: Input image side-length used by MLP input sizing.
        in_channels: Number of input image channels.

    Returns:
        Instantiated PyTorch model.
    """
    if name == "mlp":
        return MLP(int(in_channels) * int(img_size) * int(img_size), num_classes)

    local_resnet = build_local_resnet(name, num_classes, in_channels=in_channels)
    if local_resnet is not None:
        return local_resnet

    if timm is None:
        raise ValueError(f"Unknown local model '{name}' and timm is not installed")
    return timm.create_model(name, num_classes=num_classes, in_chans=int(in_channels), pretrained=False)
