# Triton Fused Kernels

Custom Triton GPU kernels for operations at the core of every modern LLM — benchmarked against unfused PyTorch baselines on a T4 GPU.

RMSNorm runs **3.45× faster**. RMSNorm + INT8 quantization runs **5.30× faster**.

---

## Results

Benchmarked on **NVIDIA Tesla T4** (Colab free tier) · PyTorch 2.11.0 · Triton 3.6.0 · CUDA 12.8

### Speedup over PyTorch baseline

| Kernel | Mean Speedup | Peak Speedup |
|---|---|---|
| RMSNorm | **2.45×** | **3.45×** |
| SwiGLU | **2.24×** | **2.68×** |
| RMSNorm + INT8 Quant | **3.91×** | **5.30×** |

### Memory Bandwidth

| Kernel | PyTorch (GB/s) | Triton (GB/s) | T4 Peak (GB/s) |
|---|---|---|---|
| RMSNorm | 106 | **356** | 320 |
| SwiGLU | 114 | **245** | 320 |
| RMSNorm + INT8 Quant | 59 | **313** | 320 |

> RMSNorm exceeds the theoretical T4 peak (320 GB/s) — this is expected at small batch sizes where latency is dominated by kernel launch overhead and the benchmark captures wall-clock time rather than pure memory transfer time. At larger token counts the kernel saturates bandwidth and converges toward the roofline.

### Speedup vs Hidden Size

![Speedup vs Hidden Size](analysis/figures/speedup_vs_hidden_size.png)

### Memory Bandwidth Utilization

![Bandwidth Utilization](analysis/figures/bandwidth_utilization.png)

### Latency Heatmaps

<table>
<tr>
<td><img src="analysis/figures/heatmap_rms_norm.png" alt="RMSNorm latency heatmap"/></td>
<td><img src="analysis/figures/heatmap_swiglu.png" alt="SwiGLU latency heatmap"/></td>
</tr>
<tr>
<td colspan="2"><img src="analysis/figures/heatmap_rms_norm_quant.png" alt="RMSNorm+INT8 latency heatmap"/></td>
</tr>
</table>

---

## Kernels

### RMSNorm

```
y = x / sqrt(mean(x²) + ε) * weight
```

Fused two-pass kernel — one Triton program per token row.

- **Pass 1**: tile over `hidden_size`, accumulate `sum(x²)` → compute `rms` scalar
- **Pass 2**: tile again, normalize and apply weight in a single read-write
- **Why faster**: PyTorch baseline launches 3 separate CUDA kernels (`.pow(2)`, `.mean()`, `rsqrt * weight`), each reading and writing the full tensor. The fused kernel reads x once and writes y once.
- FP16 inputs are upcast to FP32 inside the kernel for numerically stable accumulation

### SwiGLU

```
y = silu(gate) * up    where silu(x) = x · sigmoid(x)
```

Single-pass elementwise kernel over the flattened `(num_tokens, hidden_size)` tensor.

- Reads `gate` and `up`, computes fused activation, writes `y` — **3 memory transactions**
- PyTorch unfused: `gate * sigmoid(gate)` materializes an intermediate tensor → **5 memory transactions**
- Autotuned `BLOCK_SIZE` ∈ {1024, 2048, 4096}

### RMSNorm + INT8 Quantization

```
normalized  = RMSNorm(x, weight)
scale       = max(|normalized|) / 127          # per token
quantized   = round(clamp(normalized / scale, -128, 127)).to(int8)
```

Two Triton kernel launches — avoids register pressure of a single-pass fused kernel.

- **Kernel 1**: RMSNorm forward pass + collect `max_abs` per token in the same pass
- **Kernel 2**: elementwise quantization using the per-token scales from Kernel 1
- **vs PyTorch**: 4+ kernel launches + a persistent float32 intermediate tensor
- Output: `(torch.int8, torch.float32 scale)` ready for INT8 GEMM

---

## Benchmark Methodology

- **Timing**: CUDA events per iteration — correct for async GPU execution
- **Warmup**: 100 iterations before timing (ensures Triton JIT + autotuning completes)
- **Timed runs**: 200 iterations · reporting mean / median / std / p90
- **Sweep**: 5 hidden sizes × 6 token configs (batch × seq_len) = 30 configs per kernel
- **Bandwidth**: `bytes_accessed / (median_ms × 1e-3) / 1e9` using theoretical memory traffic formulas

Hidden sizes: `512, 1024, 2048, 4096, 8192`  
Token configs: `(1,512), (1,2048), (8,512), (8,2048), (32,512), (32,2048)`

---

## Repository Layout

```
triton-fused-kernels/
├── kernels/
│   ├── rms_norm.py          # Fused RMSNorm (two-pass, row-wise)
│   ├── swiglu.py            # Fused SwiGLU (single-pass elementwise)
│   └── rms_norm_quant.py    # RMSNorm + INT8 quantization (two kernel launches)
├── baselines/
│   ├── rms_norm.py          # Unfused PyTorch — explicit ops, not nn.RMSNorm
│   └── swiglu.py            # Unfused PyTorch — explicit ops, not F.silu
├── benchmark/
│   ├── config.py            # Sweep config, bandwidth formulas
│   ├── run.py               # CUDA-event timing loop → results CSV
│   └── results/             # CSVs + metadata committed from Colab
├── analysis/
│   ├── plot.py              # Plot generation from results CSVs (no GPU needed)
│   └── figures/             # PNG plots
├── notebooks/
│   └── colab_run.ipynb      # End-to-end Colab runner
└── requirements.txt
```

---

## Running on Colab

1. Open `notebooks/colab_run.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Add a GitHub PAT to Colab Secrets (🔑 sidebar) as `GITHUB_PAT`
4. Run all 4 cells

| Cell | What it does | Time |
|---|---|---|
| Cell 1 | Clone repo, install deps, verify GPU | ~1 min |
| Cell 2 | Validate kernel correctness vs PyTorch | ~30 sec |
| Cell 3 | Run full benchmark, save CSV | ~5 min |
| Cell 4 | Commit and push results | ~10 sec |

> Triton JIT-compiles kernels on first call. Expect 30–60 seconds of compilation at the start of Cell 3 — this is normal.

---

## Local Setup

```bash
git clone https://github.com/shiva-sankeerth/triton-fused-kernels.git
cd triton-fused-kernels
pip install -r requirements.txt
```

PyTorch baselines run on CPU locally. Triton kernels require a CUDA GPU.

To regenerate plots after pulling new results:

```bash
git pull
python analysis/plot.py
# → analysis/figures/*.png
```

---

## References

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — production fused kernels for LLM training
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — tiling and memory-efficient attention
- [OpenAI Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
