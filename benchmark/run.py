"""
Benchmark runner for fused Triton kernels vs PyTorch baselines.

Usage:
    python benchmark/run.py [--dtype float32|float16] [--output-dir benchmark/results]

Outputs:
    benchmark/results/results_YYYYMMDD_HHMMSS.csv
    benchmark/results/metadata.json
"""

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.rms_norm import rms_norm_pytorch
from baselines.swiglu import swiglu_pytorch
from benchmark.config import (
    DTYPE, DTYPE_SIZE, EPS, HIDDEN_SIZES, NUM_TIMED, NUM_WARMUP,
    TOKEN_CONFIGS, bytes_rms_norm, bytes_rms_norm_quant, bytes_swiglu,
)
from kernels.rms_norm import rms_norm_triton
from kernels.rms_norm_quant import rms_norm_quant_triton
from kernels.swiglu import swiglu_triton

CSV_FIELDS = [
    "kernel_name", "variant", "hidden_size", "num_tokens", "batch_size",
    "seq_len", "dtype", "mean_ms", "median_ms", "std_ms", "min_ms",
    "p90_ms", "bandwidth_gbs", "timestamp",
]


def time_kernel(fn, *args, num_warmup: int, num_timed: int):
    """
    Time a CUDA kernel using per-iteration CUDA events.

    Returns a dict with mean, median, std, min, p90 in milliseconds.
    CUDA events are the correct way to time GPU work — time.time() includes
    Python overhead and doesn't account for async GPU execution.
    """
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Warmup — triggers Triton JIT compilation and autotuning
    for _ in range(num_warmup):
        fn(*args)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_timed):
        start_evt.record()
        fn(*args)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))

    times_ms.sort()
    return {
        "mean_ms":   statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "std_ms":    statistics.stdev(times_ms),
        "min_ms":    times_ms[0],
        "p90_ms":    times_ms[int(0.9 * len(times_ms))],
    }


def bandwidth_gbs(bytes_accessed: int, median_ms: float) -> float:
    return bytes_accessed / (median_ms * 1e-3) / 1e9


def collect_metadata() -> dict:
    import triton
    import platform
    return {
        "gpu_name":       torch.cuda.get_device_name(0),
        "cuda_version":   torch.version.cuda,
        "torch_version":  torch.__version__,
        "triton_version": triton.__version__,
        "python_version": platform.python_version(),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "num_warmup":     NUM_WARMUP,
        "num_timed":      NUM_TIMED,
    }


def run_benchmark(dtype: torch.dtype, output_dir: Path):
    assert torch.cuda.is_available(), "CUDA required — run this on Colab/GPU"
    device = "cuda"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"results_{ts}.csv"
    meta_path = output_dir / "metadata.json"

    metadata = collect_metadata()
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"GPU: {metadata['gpu_name']}")
    print(f"Triton: {metadata['triton_version']}  |  PyTorch: {metadata['torch_version']}")
    print(f"Saving to {csv_path}\n")

    dtype_str = "float32" if dtype == torch.float32 else "float16"

    # Open CSV and write header immediately for partial-save on disconnection
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    csv_file.flush()

    benchmark_timestamp = datetime.now(timezone.utc).isoformat()

    configs = [
        (hidden_size, batch_size, seq_len)
        for hidden_size in HIDDEN_SIZES
        for (batch_size, seq_len) in TOKEN_CONFIGS
    ]

    # ------------------------------------------------------------------ #
    # RMSNorm                                                              #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("RMSNorm")
    print("=" * 60)
    for hidden_size, batch_size, seq_len in tqdm(configs, desc="rms_norm"):
        num_tokens = batch_size * seq_len

        x      = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        weight = torch.ones(hidden_size, dtype=dtype, device=device)

        nbytes = bytes_rms_norm(num_tokens, hidden_size, dtype)

        for variant, fn in [
            ("triton",  lambda: rms_norm_triton(x, weight, eps=EPS)),
            ("pytorch", lambda: rms_norm_pytorch(x, weight, eps=EPS)),
        ]:
            stats = time_kernel(fn, num_warmup=NUM_WARMUP, num_timed=NUM_TIMED)
            bw    = bandwidth_gbs(nbytes, stats["median_ms"])
            row   = {
                "kernel_name":   "rms_norm",
                "variant":       variant,
                "hidden_size":   hidden_size,
                "num_tokens":    num_tokens,
                "batch_size":    batch_size,
                "seq_len":       seq_len,
                "dtype":         dtype_str,
                "bandwidth_gbs": round(bw, 4),
                "timestamp":     benchmark_timestamp,
                **{k: round(v, 6) for k, v in stats.items()},
            }
            writer.writerow(row)
            csv_file.flush()

    # ------------------------------------------------------------------ #
    # SwiGLU                                                               #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SwiGLU")
    print("=" * 60)
    for hidden_size, batch_size, seq_len in tqdm(configs, desc="swiglu"):
        num_tokens = batch_size * seq_len

        gate = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        up   = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        nbytes = bytes_swiglu(num_tokens, hidden_size, dtype)

        for variant, fn in [
            ("triton",  lambda: swiglu_triton(gate, up)),
            ("pytorch", lambda: swiglu_pytorch(gate, up)),
        ]:
            stats = time_kernel(fn, num_warmup=NUM_WARMUP, num_timed=NUM_TIMED)
            bw    = bandwidth_gbs(nbytes, stats["median_ms"])
            row   = {
                "kernel_name":   "swiglu",
                "variant":       variant,
                "hidden_size":   hidden_size,
                "num_tokens":    num_tokens,
                "batch_size":    batch_size,
                "seq_len":       seq_len,
                "dtype":         dtype_str,
                "bandwidth_gbs": round(bw, 4),
                "timestamp":     benchmark_timestamp,
                **{k: round(v, 6) for k, v in stats.items()},
            }
            writer.writerow(row)
            csv_file.flush()

    # ------------------------------------------------------------------ #
    # RMSNorm + INT8 Quantization (float32 only — quant kernel is fp32)  #
    # ------------------------------------------------------------------ #
    if dtype == torch.float32:
        print("\n" + "=" * 60)
        print("RMSNorm + INT8 Quantization")
        print("=" * 60)
        for hidden_size, batch_size, seq_len in tqdm(configs, desc="rms_norm_quant"):
            num_tokens = batch_size * seq_len

            x      = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device)
            weight = torch.ones(hidden_size, dtype=torch.float32, device=device)

            nbytes = bytes_rms_norm_quant(num_tokens, hidden_size, torch.float32)

            # Triton fused (2 kernel launches)
            stats = time_kernel(
                lambda: rms_norm_quant_triton(x, weight, eps=EPS),
                num_warmup=NUM_WARMUP, num_timed=NUM_TIMED,
            )
            bw = bandwidth_gbs(nbytes, stats["median_ms"])
            writer.writerow({
                "kernel_name":   "rms_norm_quant",
                "variant":       "triton",
                "hidden_size":   hidden_size,
                "num_tokens":    num_tokens,
                "batch_size":    batch_size,
                "seq_len":       seq_len,
                "dtype":         "float32",
                "bandwidth_gbs": round(bw, 4),
                "timestamp":     benchmark_timestamp,
                **{k: round(v, 6) for k, v in stats.items()},
            })
            csv_file.flush()

            # PyTorch unfused baseline: rms_norm → max → clamp → round → int8
            def pytorch_rms_norm_quant():
                normed = rms_norm_pytorch(x, weight, eps=EPS)
                max_abs = normed.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = max_abs / 127.0
                q = (normed / scale).clamp(-128, 127).round().to(torch.int8)
                return q, scale.squeeze(-1)

            stats = time_kernel(
                pytorch_rms_norm_quant, num_warmup=NUM_WARMUP, num_timed=NUM_TIMED,
            )
            bw = bandwidth_gbs(nbytes, stats["median_ms"])
            writer.writerow({
                "kernel_name":   "rms_norm_quant",
                "variant":       "pytorch",
                "hidden_size":   hidden_size,
                "num_tokens":    num_tokens,
                "batch_size":    batch_size,
                "seq_len":       seq_len,
                "dtype":         "float32",
                "bandwidth_gbs": round(bw, 4),
                "timestamp":     benchmark_timestamp,
                **{k: round(v, 6) for k, v in stats.items()},
            })
            csv_file.flush()

    csv_file.close()
    print(f"\nDone. Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Triton fused kernel benchmark")
    parser.add_argument(
        "--dtype", choices=["float32", "float16"], default="float32",
        help="Input dtype (default: float32)",
    )
    parser.add_argument(
        "--output-dir", default="benchmark/results",
        help="Directory to save CSV results (default: benchmark/results)",
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    run_benchmark(dtype=dtype, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
