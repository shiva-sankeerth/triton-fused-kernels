from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,       # float32 normalized output (temp buffer)
    max_abs_ptr,   # float32 per-token max(|normalized|) output
    num_tokens,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 1 of 2: RMSNorm + per-token max(|normalized|).

    Same two-pass RMSNorm as rms_norm.py, plus accumulates max_abs in pass 2.
    Writes normalized floats to out_ptr and per-token max_abs to max_abs_ptr.
    """
    pid = tl.program_id(0)

    # Pass 1: sum of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for tile_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        x_chunk = tl.load(x_ptr + pid * hidden_size + offsets, mask=mask, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    rms = tl.sqrt(sum_sq / hidden_size + eps)

    # Pass 2: normalize, apply weight, track max_abs
    max_abs = tl.zeros([1], dtype=tl.float32)
    for tile_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size

        x_chunk = tl.load(x_ptr + pid * hidden_size + offsets, mask=mask, other=0.0)
        w_chunk = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

        x_chunk = x_chunk.to(tl.float32)
        y_chunk = x_chunk / rms * w_chunk.to(tl.float32)

        tl.store(out_ptr + pid * hidden_size + offsets, y_chunk, mask=mask)

        abs_chunk = tl.abs(y_chunk)
        # Mask out-of-bounds positions before taking max
        abs_chunk = tl.where(mask, abs_chunk, tl.zeros_like(abs_chunk))
        max_abs = tl.maximum(max_abs, tl.max(abs_chunk, axis=0))

    tl.store(max_abs_ptr + pid, max_abs)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["num_elements"],
)
@triton.jit
def _quantize_kernel(
    norm_ptr,      # float32 normalized values (num_tokens, hidden_size)
    max_abs_ptr,   # float32 per-token max_abs (num_tokens,)
    out_ptr,       # int8 output (num_tokens, hidden_size)
    num_tokens,
    hidden_size,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel 2 of 2: per-token INT8 quantization.

    scale = max_abs / 127
    q = round(clamp(normalized / scale, -128, 127))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Derive token index from flat offset to load the right per-token scale
    token_idx = offsets // hidden_size

    norm_vals = tl.load(norm_ptr + offsets, mask=mask, other=0.0)
    max_abs = tl.load(max_abs_ptr + token_idx, mask=mask, other=1.0)

    scale = max_abs / 127.0
    # Avoid division by zero for zero rows
    scale = tl.where(scale == 0.0, tl.ones_like(scale), scale)

    q = norm_vals / scale
    q = tl.clamp(q, -128.0, 127.0)
    q = tl.extra.cuda.libdevice.round(q).to(tl.int8)

    tl.store(out_ptr + offsets, q, mask=mask)


def rms_norm_quant_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton RMSNorm + per-token INT8 quantization.

    Two kernel launches (vs 4+ for the unfused PyTorch equivalent), with no
    persistent intermediate tensor for normalized values beyond the temp buffer.

    Args:
        x:      (num_tokens, hidden_size) contiguous float32 CUDA tensor
        weight: (hidden_size,) contiguous float32 CUDA tensor
        eps:    numerical stability constant

    Returns:
        quantized: (num_tokens, hidden_size) torch.int8
        scale:     (num_tokens,) torch.float32 — per-token scale = max_abs / 127
    """
    assert x.ndim == 2, f"expected 2D input, got {x.shape}"
    assert x.is_contiguous() and weight.is_contiguous(), "inputs must be contiguous"
    assert x.is_cuda, "x must be on CUDA"
    assert x.dtype == torch.float32, "rms_norm_quant only supports float32 input"

    num_tokens, hidden_size = x.shape
    num_elements = num_tokens * hidden_size

    norm_buf = torch.empty_like(x)                              # float32 temp
    max_abs  = torch.zeros(num_tokens, dtype=torch.float32, device=x.device)
    quantized = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=x.device)

    # Kernel 1: RMSNorm + collect max_abs per token
    _rms_norm_fwd_kernel[(num_tokens,)](
        x, weight, norm_buf, max_abs,
        num_tokens, hidden_size, eps,
    )

    # Kernel 2: quantize normalized values using per-token scales
    def grid_fn(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    _quantize_kernel[grid_fn](
        norm_buf, max_abs, quantized,
        num_tokens, hidden_size, num_elements,
    )

    scale = max_abs / 127.0
    return quantized, scale
