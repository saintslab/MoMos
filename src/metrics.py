import bz2
import gzip
import lzma
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pybdm\.utils",
    )
    try:
        from pybdm import BDM
    except Exception:
        BDM = None


bdm_engine = BDM(ndim=1) if BDM is not None else None

def flatten_weights(model):
    """Flatten trainable parameters into one NumPy vector.

    Args:
        model: PyTorch model to inspect.

    Returns:
        1D float32 NumPy array of trainable parameters.
    """
    arrays = []
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            arrays.append(param.detach().cpu().reshape(-1).float().numpy())
    if not arrays:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrays)


def compute_sparsity(model, compression_binarized=False):
    """Compute exact-zero fraction over trainable weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; kept for metric API consistency.

    Returns:
        Dict containing ``sparsity``.
    """
    _ = compression_binarized
    weights = flatten_weights(model)
    if weights.size == 0:
        return {"sparsity": 0.0}
    return {"sparsity": float((weights == 0).mean())}


def compute_l2(model, compression_binarized=False):
    """Compute L2 norm over trainable weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; kept for metric API consistency.

    Returns:
        Dict containing ``weight_l2``.
    """
    _ = compression_binarized
    weights = flatten_weights(model)
    if weights.size == 0:
        return {"weight_l2": 0.0}
    return {"weight_l2": float(np.linalg.norm(weights))}


def get_compression_payload(model, compression_binarized):
    """Serialize model weights to bytes for compression metrics.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, serialize sign bits only.

    Returns:
        Byte payload used by compression backends.
    """
    weights = flatten_weights(model)
    return get_compression_payload_from_weights(weights, compression_binarized=compression_binarized)


def compression_rate(payload, compressed_payload):
    """Compute compression ratio as raw_size/compressed_size.

    Args:
        payload: Uncompressed bytes.
        compressed_payload: Compressed bytes.

    Returns:
        Float compression ratio (0.0 for empty payload).
    """
    raw_size = len(payload)
    if raw_size == 0:
        return 0.0
    compressed_size = max(1, len(compressed_payload))
    return float(raw_size / compressed_size)


def compute_gzip(model, compression_binarized=False):
    """Compute gzip compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``gzip_compression_rate``.
    """
    payload = get_compression_payload(model, compression_binarized)
    return {"gzip_compression_rate": compression_rate(payload, gzip.compress(payload, compresslevel=9, mtime=0))}


def compute_bz2(model, compression_binarized=False):
    """Compute bz2 compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``bz2_compression_rate``.
    """
    payload = get_compression_payload(model, compression_binarized)
    return {"bz2_compression_rate": compression_rate(payload, bz2.compress(payload, compresslevel=9))}


def compute_lzma(model, compression_binarized=False):
    """Compute lzma compression rate of model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: If True, compress sign bits only.

    Returns:
        Dict containing ``lzma_compression_rate``.
    """
    payload = get_compression_payload(model, compression_binarized)
    return {"lzma_compression_rate": compression_rate(payload, lzma.compress(payload, preset=9))}


def compute_bdm(model, compression_binarized=False):
    """Compute BDM complexity on binarized model weights.

    Args:
        model: PyTorch model to inspect.
        compression_binarized: Unused; BDM always uses binarized payload.

    Returns:
        Dict containing ``bdm_complexity`` (or ``None`` if unavailable).
    """
    _ = compression_binarized
    if bdm_engine is None:
        return {"bdm_complexity": None}
    weights = flatten_weights(model)
    if weights.size == 0:
        return {"bdm_complexity": 0.0}
    bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
    try:
        complexity = float(bdm_engine.bdm(bits))
    except Exception:
        return {"bdm_complexity": None}
    return {"bdm_complexity": complexity}


registry = {
    "sparsity": compute_sparsity,
    "l2": compute_l2,
    "bdm": compute_bdm,
    "gzip": compute_gzip,
    "bz2": compute_bz2,
    "lzma": compute_lzma,
}


def register_metric(name, fn):
    """Register or replace a metric callback in the registry.

    Args:
        name: Metric name key used by ``compute_metrics``.
        fn: Callable with signature ``fn(model, compression_binarized=False)``.
    """
    registry[str(name)] = fn


def _metric_from_weights(name, weights):
    """Compute a metric directly from pre-flattened weights."""
    if name == "sparsity":
        if weights.size == 0:
            return {"sparsity": 0.0}
        return {"sparsity": float((weights == 0).mean())}
    if name == "l2":
        if weights.size == 0:
            return {"weight_l2": 0.0}
        return {"weight_l2": float(np.linalg.norm(weights))}
    if name == "bdm":
        if bdm_engine is None:
            return {"bdm_complexity": None}
        if weights.size == 0:
            return {"bdm_complexity": 0.0}
        bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
        try:
            return {"bdm_complexity": float(bdm_engine.bdm(bits))}
        except Exception:
            return {"bdm_complexity": None}
    raise KeyError(name)


def compute_metrics(model, names, compression_binarized=False):
    """Compute selected metrics on the current model weights.

    Args:
        model: PyTorch model to inspect.
        names: List of metric names to compute.
        compression_binarized: Global flag for compression metric payload format.

    Returns:
        Dict of merged metric outputs.
    """
    unknown = [name for name in names if name not in registry]
    if unknown:
        available = ", ".join(sorted(registry))
        unknown_str = ", ".join(sorted(set(str(name) for name in unknown)))
        raise ValueError(f"Unknown metric(s): {unknown_str}. Available: {available}")

    out = {}
    weights = None
    payload_cache = {}
    simple_names = {"sparsity", "l2", "bdm"}
    compression_names = {"gzip", "bz2", "lzma"}

    for name in names:
        if name in simple_names:
            if weights is None:
                weights = flatten_weights(model)
            out.update(_metric_from_weights(name, weights))
            continue

        if name in compression_names:
            if weights is None:
                weights = flatten_weights(model)
            key = bool(compression_binarized)
            if key not in payload_cache:
                payload_cache[key] = get_compression_payload_from_weights(weights, compression_binarized=key)
            payload = payload_cache[key]
            if name == "gzip":
                out["gzip_compression_rate"] = compression_rate(payload, gzip.compress(payload, compresslevel=9, mtime=0))
            elif name == "bz2":
                out["bz2_compression_rate"] = compression_rate(payload, bz2.compress(payload, compresslevel=9))
            else:
                out["lzma_compression_rate"] = compression_rate(payload, lzma.compress(payload, preset=9))
            continue

        fn = registry.get(name)
        out.update(fn(model, compression_binarized=compression_binarized))
    return out


def get_compression_payload_from_weights(weights, compression_binarized):
    """Serialize an already flattened weight vector to bytes."""
    if weights.size == 0:
        return b""
    if compression_binarized:
        bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
        return bits.tobytes()
    values = np.ascontiguousarray(weights.astype(np.float32))
    return values.tobytes()
