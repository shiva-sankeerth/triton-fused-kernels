import torch

# Hidden dimensions covering common LLM sizes
# LLaMA-7B: 4096, LLaMA-13B: 5120, Mistral-7B: 4096, Qwen-7B: 4096
HIDDEN_SIZES = [512, 1024, 2048, 4096, 8192]

# (batch_size, seq_len) pairs — collapsed to num_tokens = batch * seq_len
# RMSNorm and SwiGLU operate per-token, so only num_tokens matters for kernel perf
TOKEN_CONFIGS = [
    (1,  512),
    (1,  2048),
    (8,  512),
    (8,  2048),
    (32, 512),
    (32, 2048),
]

# Primary dtype — T4 has no native BF16; use FP32 as default, FP16 as variant
DTYPE = torch.float32

# Timing — warmup must cover Triton autotuning (first call per hidden_size config)
NUM_WARMUP = 100
NUM_TIMED  = 200

EPS = 1e-6

# T4 theoretical peak memory bandwidth (GB/s)
T4_PEAK_BANDWIDTH_GBS = 320.0

# Bytes per element for bandwidth calculation
DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
}

def bytes_rms_norm(num_tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
    """Theoretical bytes accessed by a fused RMSNorm kernel.

    Pass 1: read x once (sum-of-squares)
    Pass 2: read x + weight, write y
    Total: 3 * N * H + H (weight is small, counts once)
    """
    e = DTYPE_SIZE[dtype]
    return (3 * num_tokens * hidden_size + hidden_size) * e


def bytes_swiglu(num_tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
    """Theoretical bytes accessed by a fused SwiGLU kernel.

    One pass: read gate + up, write out
    Total: 3 * N * H
    """
    e = DTYPE_SIZE[dtype]
    return 3 * num_tokens * hidden_size * e


def bytes_rms_norm_quant(num_tokens: int, hidden_size: int, dtype: torch.dtype) -> int:
    """Theoretical bytes accessed by the fused RMSNorm+INT8 quant kernels.

    Kernel 1 (rms_norm): 3 * N * H (fp32) + H (weight) + N (max_abs write)
    Kernel 2 (quantize): N * H (fp32 normalized read) + N * H (int8 write)
    """
    e = DTYPE_SIZE[dtype]
    k1 = (3 * num_tokens * hidden_size + hidden_size + num_tokens) * e
    k2 = num_tokens * hidden_size * e + num_tokens * hidden_size * 1  # int8 = 1 byte
    return k1 + k2
