import time

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


# Target temporary size for pairwise distance chunks (~4096 MB).
DEFAULT_CHUNK_SIZE_MB = 4096


class RoundSTE(torch.autograd.Function):
    """Straight-through rounding estimator used by fake quantization."""

    @staticmethod
    def forward(ctx, input_tensor):
        return torch.round(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class UniformSymmetric(nn.Module):
    """Per-tensor signed symmetric fake quantizer."""

    def __init__(self, bitwidth=8):
        super().__init__()
        self.bitwidth = int(bitwidth)

    def forward(self, weight):
        bits = int(self.bitwidth)
        if bits >= 32:
            return weight
        if bits < 2:
            raise ValueError(f"bitwidth must be >= 2, got {bits}")
        q = 2 ** (bits - 1) - 1
        with torch.no_grad():
            absmax = torch.max(torch.abs(weight))
        if absmax <= 0:
            return weight
        scale = absmax / q
        return scale * torch.clamp(RoundSTE.apply(weight / scale), -q, q)


class FakeQuantParametrization(nn.Module):
    """Parametrization wrapper that applies fake quant in forward pass."""

    def __init__(self, quantizer, enabled=True):
        super().__init__()
        self.quantizer = quantizer
        self.enabled = bool(enabled)

    def forward(self, weight):
        if not self.enabled:
            return weight
        return self.quantizer(weight)


def _has_fake_quant_weight_parametrization(module):
    if not hasattr(module, "parametrizations"):
        return False
    plist = getattr(module.parametrizations, "weight", None)
    if plist is None:
        return False
    return any(isinstance(p, FakeQuantParametrization) for p in plist)


def attach_weight_quantizers(model, bitwidth, exclude_layers=None, enabled=True):
    """Attach or update fake-quant parametrizations on trainable layer weights.

    Only module weights are quantized (not biases). Exclusion is name-based
    and controlled by ``exclude_layers`` tokens.

    Args:
        model: Model to modify in-place.
        bitwidth: Quantization bit-width.
        exclude_layers: Optional list of substrings; matching module names are
            skipped.
        enabled: Whether attached fake quantizers are active.

    Returns:
        Dict with ``attached_modules`` and ``updated_modules`` counts.
    """
    bits = int(bitwidth)
    excludes = tuple(str(item).lower() for item in (exclude_layers or []))
    attached = 0
    updated = 0

    for name, module in model.named_modules():
        lname = str(name).lower()
        if excludes and any(token in lname for token in excludes):
            continue

        weight = getattr(module, "weight", None)
        if not isinstance(weight, nn.Parameter):
            continue
        if not weight.requires_grad or weight.numel() == 0:
            continue

        if _has_fake_quant_weight_parametrization(module):
            for p in module.parametrizations.weight:
                if isinstance(p, FakeQuantParametrization):
                    p.quantizer.bitwidth = bits
                    p.enabled = bool(enabled)
                    updated += 1
                    break
            continue

        if parametrize.is_parametrized(module, "weight"):
            # Leave non-fake parametrizations untouched.
            continue

        parametrize.register_parametrization(
            module,
            "weight",
            FakeQuantParametrization(
                quantizer=UniformSymmetric(bitwidth=bits),
                enabled=enabled,
            ),
        )
        attached += 1

    return {"attached_modules": int(attached), "updated_modules": int(updated)}


def toggle_quantization(model, enabled):
    """Enable/disable attached fake quantizers."""
    toggled = 0
    enabled = bool(enabled)
    for module in model.modules():
        if not _has_fake_quant_weight_parametrization(module):
            continue
        for p in module.parametrizations.weight:
            if isinstance(p, FakeQuantParametrization):
                p.enabled = enabled
                toggled += 1
    return int(toggled)


def prepare_qat(model, quant_cfg):
    """Configure QAT fake quantizers on a model once before training."""
    bits = int(quant_cfg.get("q", 32))
    if bits < 2:
        raise ValueError(f"q must be >= 2, got {bits}")
    if bits >= 32:
        disabled = toggle_quantization(model, False)
        return {"qat_enabled": False, "q_bits": bits, "disabled_modules": int(disabled)}

    exclude_layers = quant_cfg.get("exclude_layers", [])
    attach_stats = attach_weight_quantizers(
        model,
        bitwidth=bits,
        exclude_layers=exclude_layers,
        enabled=True,
    )
    attach_stats["qat_enabled"] = True
    attach_stats["q_bits"] = bits
    return attach_stats


def iter_trainable_params(model):
    """Yield non-empty trainable parameters from a model.

    Args:
        model: Model to iterate.

    Yields:
        Parameter tensors with ``requires_grad=True``.
    """
    for param in model.parameters():
        if param.requires_grad and param.numel() > 0:
            yield param


def tensor_to_blocks(tensor, block_size):
    """Flatten and pad a tensor into fixed-size blocks.

    Args:
        tensor: Source parameter tensor.
        block_size: Number of values per block.

    Returns:
        Tuple ``(blocks, original_numel, original_shape)``.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    flat = tensor.flatten()
    n_params = flat.numel()
    pad = (-n_params) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    return flat.view(-1, block_size), n_params, tensor.shape


def blocks_to_tensor(blocks, n_params, shape):
    """Reconstruct original tensor shape from flattened blocks.

    Args:
        blocks: Block matrix.
        n_params: Number of original (unpadded) values.
        shape: Original tensor shape.

    Returns:
        Tensor restored to original shape.
    """
    return blocks.flatten()[:n_params].view(shape)


def count_total_blocks(model, block_size):
    """Count quantization blocks across all trainable parameters.

    Args:
        model: Model to inspect.
        block_size: Number of values per block.

    Returns:
        Integer number of blocks.
    """
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    total = 0
    for param in iter_trainable_params(model):
        total += (param.numel() + block_size - 1) // block_size
    return int(total)


def k_from_capacity(model, block_size, capacity):
    """Convert capacity ratio to absolute motif count ``k``.

    Args:
        model: Model used to determine total block count.
        block_size: Number of values per block.
        capacity: Fraction of blocks to keep as motifs.

    Returns:
        Integer ``k`` clamped to ``[1, total_blocks]``.
    """
    capacity = float(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    n_blocks = count_total_blocks(model, block_size)
    k = int(capacity * n_blocks)
    return max(1, min(k, n_blocks))


def _resolve_chunk_size_blocks(n_blocks, n_motifs, chunk_size=None, dtype=torch.float32):
    """Resolve chunk size (in blocks) for block-vs-motif distance computation.

    Args:
        n_blocks: Number of blocks being assigned.
        n_motifs: Number of motifs compared against.
        chunk_size: Optional memory budget in MB for the distance matrix.
            Default is ``4096`` MB (~4 GB) when ``None``.
        dtype: Floating dtype used in distance matrix.

    Returns:
        Integer chunk size in ``[1, n_blocks]``.
    """
    n_blocks = int(n_blocks)
    n_motifs = max(1, int(n_motifs))
    if n_blocks <= 0:
        return 1

    chunk_size_mb = DEFAULT_CHUNK_SIZE_MB if chunk_size is None else float(chunk_size)
    if chunk_size_mb <= 0:
        raise ValueError(f"chunk_size must be > 0 MB, got {chunk_size}")
    bytes_budget = int(chunk_size_mb * 1024 * 1024)
    bytes_per = int(torch.empty((), dtype=dtype).element_size())
    max_elems = max(1, bytes_budget // max(1, bytes_per))
    chunk = max(1, max_elems // n_motifs)
    return min(n_blocks, chunk)


def _resolve_progress_every_elements(total_elements, progress_every_elements=None):
    """Resolve how often progress should be printed by element count.

    Args:
        total_elements: Total scalar elements processed in one call.
        progress_every_elements: Optional explicit reporting interval.

    Returns:
        Positive integer reporting interval in elements.
    """
    total_elements = int(total_elements)
    if total_elements <= 0:
        return 1
    if progress_every_elements is not None:
        value = int(progress_every_elements)
        if value <= 0:
            raise ValueError(f"progress_every_elements must be > 0, got {progress_every_elements}")
        return value
    # Default: about 20 progress updates per call.
    return max(1, total_elements // 20)


def _nearest_motifs_chunked(
    blocks,
    motifs,
    chunk_size=None,
    show_progress=False,
    progress_prefix="momos",
    progress_every_elements=None,
):
    """Assign each block to nearest motif using chunked pairwise distances.

    Args:
        blocks: Block matrix with shape ``[num_blocks, block_size]``.
        motifs: Motif matrix with shape ``[k_eff, block_size]``.
        chunk_size: Optional memory budget in MB for chunked distance
            computation. Default is ``4096`` MB (~4 GB).
        show_progress: If True, print coarse chunk progress updates.
        progress_prefix: Label prefix for progress lines.
        progress_every_elements: Optional progress print interval measured in
            processed scalar elements.

    Returns:
        Long tensor of nearest motif indices for each block.
    """
    n_blocks = int(blocks.size(0))
    n_motifs = int(motifs.size(0))
    chunk = _resolve_chunk_size_blocks(
        n_blocks,
        n_motifs,
        chunk_size=chunk_size,
        dtype=blocks.dtype,
    )

    nearest = torch.empty(n_blocks, dtype=torch.long, device=blocks.device)
    motifs_t = motifs.t().contiguous()
    motifs_norm2 = (motifs * motifs).sum(dim=1).view(1, -1)
    elements_per_block = int(blocks.size(1)) if blocks.dim() > 1 else 1
    total_elements = n_blocks * elements_per_block
    total_chunks = (n_blocks + chunk - 1) // chunk
    print_every = _resolve_progress_every_elements(
        total_elements,
        progress_every_elements=progress_every_elements,
    )
    next_emit = print_every
    for start in range(0, n_blocks, chunk):
        end = min(start + chunk, n_blocks)
        chunk_blocks = blocks[start:end]
        chunk_norm2 = (chunk_blocks * chunk_blocks).sum(dim=1, keepdim=True)
        chunk_idx = (start // chunk) + 1
        # Exact Euclidean argmin via squared distances:
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        dist2 = chunk_norm2 + motifs_norm2 - 2.0 * chunk_blocks.matmul(motifs_t)
        nearest[start:end] = dist2.argmin(dim=1)

        if show_progress:
            local_done = end * elements_per_block
            should_emit = local_done >= next_emit or end == n_blocks
            if should_emit:
                pct = 100.0 * float(local_done) / float(max(1, total_elements))
                blocks_done = int(end)
                blocks_total = int(n_blocks)
                pairwise_done = int(blocks_done * n_motifs)
                pairwise_total = int(blocks_total * n_motifs)
                print(
                    f"{progress_prefix}: chunk {chunk_idx}/{total_chunks} "
                    f"blocks {blocks_done:,}/{blocks_total:,} "
                    f"pairwise {pairwise_done:,}/{pairwise_total:,} ({pct:.1f}%)",
                    flush=True,
                )
                while next_emit <= local_done:
                    next_emit += print_every
    return nearest


def momos(
    model,
    block_size,
    k,
    force_zero=True,
    chunk_size=None,
    show_chunk_progress=False,
    progress_prefix="momos",
    progress_every_elements=None,
):
    """Apply one MoMos projection step.

    Args:
        model: Model to quantize in-place.
        block_size: Block size ``s``.
        k: Number of motifs.
        force_zero: If True, include all-zero motif.
        chunk_size: Optional memory budget in MB for nearest-motif distance
            chunking. ``None`` uses the default ``4096`` MB (~4 GB).
        show_chunk_progress: If True, print coarse chunk progress updates.
        progress_prefix: Label prefix for chunk progress lines.
        progress_every_elements: Optional progress print interval measured in
            processed scalar elements.

    Motifs are selected globally across all trainable blocks, then each
    block is assigned to its nearest motif (also globally).

    Returns:
        Dict with distortion, changed weights, and motif usage counts.
    """
    block_size = int(block_size)
    k = int(k)
    motif_counts = torch.zeros(max(1, k), dtype=torch.long)

    with torch.no_grad():
        params = list(iter_trainable_params(model))
        layer_specs = []
        all_blocks = []

        for param in params:
            blocks, n_params, shape = tensor_to_blocks(param.data, block_size)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)

        if not all_blocks:
            return {
                "distortion": 0.0,
                "num_changed_weights": 0,
                "motif_counts": motif_counts,
            }

        all_blocks = torch.cat(all_blocks, dim=0)
        total_blocks = int(all_blocks.size(0))
        k_eff = max(1, min(k, total_blocks))

        if force_zero:
            if k_eff == 1:
                motifs = torch.zeros(1, block_size, device=all_blocks.device, dtype=all_blocks.dtype)
            else:
                idx = torch.randperm(total_blocks, device=all_blocks.device)[: k_eff - 1]
                motifs = torch.cat([torch.zeros(1, block_size, device=all_blocks.device, dtype=all_blocks.dtype), all_blocks[idx]], dim=0)
        else:
            idx = torch.randperm(total_blocks, device=all_blocks.device)[:k_eff]
            motifs = all_blocks[idx]

        if force_zero and k_eff == 1:
            nearest = torch.zeros(total_blocks, dtype=torch.long, device=all_blocks.device)
            quantized_blocks = motifs.expand(total_blocks, -1)
        else:
            nearest = _nearest_motifs_chunked(
                all_blocks,
                motifs,
                chunk_size=chunk_size,
                show_progress=bool(show_chunk_progress),
                progress_prefix=str(progress_prefix),
                progress_every_elements=progress_every_elements,
            )
            quantized_blocks = motifs[nearest]

        counts = torch.bincount(nearest, minlength=k_eff).to("cpu", dtype=torch.long)
        motif_counts[:k_eff] = counts

        distortion = 0.0
        changed_weights = 0
        offset = 0
        for param, n_blocks, n_params, shape in layer_specs:
            next_offset = offset + n_blocks
            src_blocks = all_blocks[offset:next_offset]
            q_blocks = quantized_blocks[offset:next_offset]

            # Reconstruct and compare only real (unpadded) entries for metrics.
            src_flat = src_blocks.flatten()[:n_params]
            q_flat = q_blocks.flatten()[:n_params]
            distortion += float((src_flat - q_flat).pow(2).sum().item())
            changed_weights += int((src_flat != q_flat).sum().item())

            param.data.copy_(blocks_to_tensor(q_blocks, n_params, shape))
            offset = next_offset

    return {
        "distortion": float(distortion),
        "num_changed_weights": int(changed_weights),
        "motif_counts": motif_counts,
    }


def quantize_qat(model, quant_cfg):
    """Prepare fake-quant QAT parametrizations.

    Args:
        model: Model to quantize.
        quant_cfg: Config dict containing at least ``q``.

    Returns:
        Dict describing QAT setup/update state.
    """
    return prepare_qat(model, quant_cfg)


def quantize_momos(model, quant_cfg):
    """Apply one MoMos projection step and return projection stats."""
    out = momos(
        model,
        quant_cfg["s"],
        quant_cfg["k"],
        force_zero=quant_cfg.get("force_zero", False),
        chunk_size=quant_cfg.get("chunk_size"),
        show_chunk_progress=quant_cfg.get("chunk_progress", False),
        progress_prefix=quant_cfg.get("progress_prefix", "computing nearest motifs"),
        progress_every_elements=quant_cfg.get("chunk_progress_elements"),
    )
    return out


METHODS = {
    "qat": quantize_qat,
    "momos": quantize_momos,
}


def available_methods():
    """Return supported quantization method names."""
    return sorted(METHODS.keys())


class MoMos:
    """Callable helper that applies MoMos using stored settings."""

    def __init__(
        self,
        s,
        k=None,
        capacity=None,
        q=32,
        force_zero=True,
        chunk_size=None,
        chunk_progress=False,
        chunk_progress_elements=None,
    ):
        """Store MoMos hyperparameters.

        Args:
            s: Block size.
            k: Optional motif count.
            capacity: Optional ratio used to derive motif count.
            q: Optional QAT bit-width metadata (used by callers that also
                enable fake-quant QAT separately).
            force_zero: Whether to force inclusion of zero motif.
            chunk_size: Optional memory budget in MB for nearest-motif distance
                chunking. Default is ``4096`` MB (~4 GB).
            chunk_progress: If True, print coarse chunk progress updates.
            chunk_progress_elements: Optional progress print interval measured
                in processed scalar elements.
        """
        self.s = int(s)
        self.k = None if k is None else int(k)
        self.capacity = capacity
        self.q = int(q)
        self.force_zero = bool(force_zero)
        self.chunk_size = None if chunk_size is None else float(chunk_size)
        self.chunk_progress = bool(chunk_progress)
        self.chunk_progress_elements = None if chunk_progress_elements is None else int(chunk_progress_elements)

    def resolve_k(self, model):
        """Resolve motif count directly or from capacity.

        Args:
            model: Model used when ``capacity`` is provided.

        Returns:
            Integer motif count ``k``.
        """
        if self.k is not None:
            return self.k
        if self.capacity is None:
            raise ValueError("MoMos requires k or capacity")
        return k_from_capacity(model, self.s, self.capacity)

    def config(self, model):
        """Build quantizer config dict for a given model.

        Args:
            model: Model used for resolving ``k`` when needed.

        Returns:
            Config dictionary consumed by ``quantize_momos``.
        """
        return {
            "method": "momos",
            "s": self.s,
            "k": self.resolve_k(model),
            "q": self.q,
            "force_zero": self.force_zero,
            "capacity": self.capacity,
            "chunk_size": self.chunk_size,
            "chunk_progress": self.chunk_progress,
            "chunk_progress_elements": self.chunk_progress_elements,
        }

    def __call__(self, model):
        """Apply MoMos once and return quantization stats.

        Args:
            model: Model to quantize.

        Returns:
            Stats dict including method name.
        """
        out = quantize_momos(model, self.config(model))
        out["method"] = "momos"
        return out


def quantize(model, quant_cfg):
    """Dispatch quantization by method and attach elapsed time.

    Args:
        model: Model to quantize.
        quant_cfg: Quantization config dict with ``method`` key.

    Returns:
        Stats dict with ``method`` and ``q_time``, or ``None``.
    """
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
