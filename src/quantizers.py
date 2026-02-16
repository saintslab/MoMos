import time

import torch


def iter_trainable_params(model):
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            yield param


def tensor_to_blocks(tensor, block_size):
    flat = tensor.flatten()
    n_params = flat.numel()
    pad = (-n_params) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    return flat.view(-1, block_size), n_params, tensor.shape


def blocks_to_tensor(blocks, n_params, shape):
    return blocks.flatten()[:n_params].view(shape)


def count_total_blocks(model, block_size):
    block_size = int(block_size)
    total = 0
    for param in iter_trainable_params(model):
        total += (param.numel() + block_size - 1) // block_size
    return int(total)


def k_from_capacity(model, block_size, capacity):
    capacity = float(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    n_blocks = count_total_blocks(model, block_size)
    k = int(capacity * n_blocks)
    return max(1, min(k, n_blocks))


def momos(model, block_size, k, force_zero=True):
    block_size = int(block_size)
    k = int(k)
    distortion = 0.0
    changed_weights = 0
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    for param in iter_trainable_params(model):
        blocks, n_params, shape = tensor_to_blocks(param.data, block_size)
        n_blocks = blocks.size(0)
        k_eff = max(1, min(k, n_blocks))

        if force_zero:
            if k_eff == 1:
                motifs = torch.zeros(1, block_size, device=blocks.device, dtype=blocks.dtype)
            else:
                idx = torch.randperm(n_blocks, device=blocks.device)[: k_eff - 1]
                motifs = torch.cat([torch.zeros(1, block_size, device=blocks.device, dtype=blocks.dtype), blocks[idx]], dim=0)
        else:
            idx = torch.randperm(n_blocks, device=blocks.device)[:k_eff]
            motifs = blocks[idx]

        nearest = torch.cdist(blocks, motifs).argmin(dim=1)
        quantized_blocks = motifs[nearest]
        counts = torch.bincount(nearest, minlength=k_eff).to("cpu", dtype=torch.long)
        motif_counts[:k_eff] += counts

        distortion += (blocks - quantized_blocks).pow(2).sum().item()
        changed_weights += (blocks != quantized_blocks).sum().item()
        param.data.copy_(blocks_to_tensor(quantized_blocks, n_params, shape))

    return {
        "distortion": distortion,
        "num_changed_weights": int(changed_weights),
        "motif_counts": motif_counts,
    }


def apply_precision_projection(model, bits):
    bits = int(bits)
    if bits >= 32:
        return
    q = 2 ** (bits - 1) - 1
    for param in iter_trainable_params(model):
        weights = param.data
        max_abs = weights.abs().max()
        if max_abs == 0:
            continue
        scale = max_abs / q
        param.data.copy_(torch.round(weights / scale).clamp(-q, q) * scale)
    return


def quantize_qat(model, quant_cfg):
    apply_precision_projection(model, quant_cfg.get("q", 32))
    return {}


def quantize_momos(model, quant_cfg):
    out = momos(
        model,
        quant_cfg["s"],
        quant_cfg["k"],
        force_zero=quant_cfg.get("force_zero", False),
    )
    apply_precision_projection(model, quant_cfg.get("q", 32))
    return out


METHODS = {
    "qat": quantize_qat,
    "momos": quantize_momos,
}


def available_methods():
    return sorted(METHODS.keys())


class MoMos:
    def __init__(self, s, k=None, capacity=None, q=32, force_zero=True):
        self.s = int(s)
        self.k = None if k is None else int(k)
        self.capacity = capacity
        self.q = int(q)
        self.force_zero = bool(force_zero)

    def resolve_k(self, model):
        if self.k is not None:
            return self.k
        if self.capacity is None:
            raise ValueError("MoMos requires k or capacity")
        return k_from_capacity(model, self.s, self.capacity)

    def config(self, model):
        return {
            "method": "momos",
            "s": self.s,
            "k": self.resolve_k(model),
            "q": self.q,
            "force_zero": self.force_zero,
            "capacity": self.capacity,
        }

    def __call__(self, model):
        out = quantize_momos(model, self.config(model))
        out["method"] = "momos"
        return out


def quantize(model, quant_cfg):
    if not quant_cfg:
        return None

    start = time.perf_counter()
    method = str(quant_cfg.get("method", "qat")).lower()
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unsupported quantization method: {method}. Available: {', '.join(available_methods())}")
    out = fn(model, quant_cfg) or {}
    out["method"] = method
    out["q_time"] = time.perf_counter() - start
    return out
