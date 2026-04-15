import torch


def rms_norm_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Unfused PyTorch RMSNorm reference.

    Explicit formula (not nn.RMSNorm) so numerics match the Triton kernel exactly.
    Runs three CUDA kernels: pow, mean, rsqrt+mul.

    Args:
        x:      (num_tokens, hidden_size) float32 or float16
        weight: (hidden_size,) float32 or float16
        eps:    numerical stability constant

    Returns:
        (num_tokens, hidden_size) same dtype as x
    """
    assert x.ndim == 2, f"expected 2D input, got shape {x.shape}"
    variance = x.pow(2).mean(dim=-1, keepdim=True)   # (num_tokens, 1)
    x_norm = x * torch.rsqrt(variance + eps)          # (num_tokens, hidden_size)
    return x_norm * weight                             # broadcast (hidden_size,)
