import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["num_elements"],
)
@triton.jit
def _swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SwiGLU kernel.

    Reads gate and up in a single pass, computes silu(gate) * up, writes output.
    Saves one full read + one full write vs unfused (no intermediate silu tensor).

    Memory traffic: 3 * N * H * element_size (read gate, read up, write out)
    Unfused baseline: 5 * N * H * element_size (silu writes intermediate, mul re-reads it)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)

    # silu(x) = x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(out_ptr + offsets, out, mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused Triton SwiGLU: y = silu(gate) * up

    Args:
        gate: (num_tokens, hidden_size) contiguous float32 or float16 CUDA tensor
        up:   (num_tokens, hidden_size) same shape, dtype, and device as gate

    Returns:
        (num_tokens, hidden_size) same dtype as gate
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    assert gate.is_contiguous() and up.is_contiguous(), "inputs must be contiguous"
    assert gate.is_cuda, "tensors must be on CUDA"

    num_elements = gate.numel()
    out = torch.empty_like(gate)

    grid = (triton.cdiv(num_elements, 1),)  # autotuner determines BLOCK_SIZE

    # Grid is a lambda so autotuner can read BLOCK_SIZE from the chosen config
    def grid_fn(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    _swiglu_kernel[grid_fn](gate, up, out, num_elements)
    return out
