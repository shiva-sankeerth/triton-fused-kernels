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
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    num_tokens,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm forward kernel.

    One program per token (row). Two passes over the hidden dimension:
      Pass 1 — accumulate sum of squares (keeps x in L2 between passes for small hidden)
      Pass 2 — normalize and apply weight

    FP16 inputs are cast to FP32 for accumulation to avoid precision loss
    when summing squared values across large hidden dimensions.
    """
    pid = tl.program_id(0)  # token index

    # Pass 1: accumulate sum of x^2 across tiles
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for tile_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        x_chunk = tl.load(x_ptr + pid * hidden_size + offsets, mask=mask, other=0.0)
        x_chunk = x_chunk.to(tl.float32)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    rms = tl.sqrt(sum_sq / hidden_size + eps)  # scalar

    # Pass 2: normalize and apply weight
    for tile_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size

        x_chunk = tl.load(x_ptr + pid * hidden_size + offsets, mask=mask, other=0.0)
        w_chunk = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

        x_chunk = x_chunk.to(tl.float32)
        y_chunk = x_chunk / rms * w_chunk.to(tl.float32)

        # Cast back to input dtype before storing
        y_chunk = y_chunk.to(x_chunk.dtype)
        tl.store(y_ptr + pid * hidden_size + offsets, y_chunk, mask=mask)


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused Triton RMSNorm.

    Args:
        x:      (num_tokens, hidden_size) contiguous float32 or float16 CUDA tensor
        weight: (hidden_size,) contiguous float32 or float16 CUDA tensor
        eps:    numerical stability constant

    Returns:
        (num_tokens, hidden_size) same dtype and device as x
    """
    assert x.ndim == 2, f"expected 2D input, got {x.shape}"
    assert x.is_contiguous(), "x must be contiguous — call .contiguous() first"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert x.is_cuda, "x must be on CUDA"

    num_tokens, hidden_size = x.shape
    y = torch.empty_like(x)

    grid = (num_tokens,)
    _rms_norm_kernel[grid](
        x, weight, y,
        num_tokens, hidden_size, eps,
    )
    return y
