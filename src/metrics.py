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
    arrays = []
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            arrays.append(param.detach().cpu().reshape(-1).float().numpy())
    if not arrays:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrays)


def compute_sparsity(model, compression_binarized=False):
    weights = flatten_weights(model)
    if weights.size == 0:
        return {"sparsity": 0.0}
    return {"sparsity": float((weights == 0).mean())}


def compute_l2(model, compression_binarized=False):
    weights = flatten_weights(model)
    if weights.size == 0:
        return {"weight_l2": 0.0}
    return {"weight_l2": float(np.linalg.norm(weights))}


def get_compression_payload(model, compression_binarized):
    weights = flatten_weights(model)
    if weights.size == 0:
        return b""
    if compression_binarized:
        bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
        return bits.tobytes()
    weights = np.ascontiguousarray(weights.astype(np.float32))
    return weights.tobytes()


def compression_rate(payload, compressed_payload):
    raw_size = len(payload)
    if raw_size == 0:
        return 0.0
    compressed_size = max(1, len(compressed_payload))
    return float(raw_size / compressed_size)


def compute_gzip(model, compression_binarized=False):
    payload = get_compression_payload(model, compression_binarized)
    return {"gzip_compression_rate": compression_rate(payload, gzip.compress(payload, compresslevel=9, mtime=0))}


def compute_bz2(model, compression_binarized=False):
    payload = get_compression_payload(model, compression_binarized)
    return {"bz2_compression_rate": compression_rate(payload, bz2.compress(payload, compresslevel=9))}


def compute_lzma(model, compression_binarized=False):
    payload = get_compression_payload(model, compression_binarized)
    return {"lzma_compression_rate": compression_rate(payload, lzma.compress(payload, preset=9))}


def compute_bdm(model, compression_binarized=False):
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
    registry[str(name)] = fn


def compute_metrics(model, names, compression_binarized=False):
    out = {}
    for name in names:
        fn = registry.get(name)
        if fn is None:
            continue
        if name == "bdm":
            out.update(fn(model, compression_binarized=True))
        else:
            out.update(fn(model, compression_binarized=compression_binarized))
    return out
