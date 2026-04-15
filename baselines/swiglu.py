import torch


def swiglu_pytorch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Unfused PyTorch SwiGLU reference.

    Deliberately uses explicit sigmoid instead of F.silu to guarantee two
    separate CUDA kernels (silu then multiply), making the baseline truly
    unfused for a fair bandwidth comparison.

    Formula: y = (gate * sigmoid(gate)) * up

    Args:
        gate: (num_tokens, hidden_size)
        up:   (num_tokens, hidden_size) same shape and dtype as gate

    Returns:
        (num_tokens, hidden_size) same dtype as gate
    """
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"
    silu_gate = gate * torch.sigmoid(gate)   # kernel 1: silu
    return silu_gate * up                    # kernel 2: elementwise mul
