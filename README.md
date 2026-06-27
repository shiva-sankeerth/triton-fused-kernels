# Triton Fused Kernels

Benchmarking fused Triton GPU kernels against unfused PyTorch baselines for operations at the core of every modern LLM (LLaMA, Mistral, Qwen, Gemma).

| Kernel | What it does | Why it matters |
|---|---|---|
| **RMSNorm** | Fused two-pass row-wise normalization | Every transformer layer boundary |
| **SwiGLU** | Fused gate activation + multiply | FFN in LLaMA / Mistral / Qwen |
| **RMSNorm + INT8** | RMSNorm with per-token INT8 quantization | Quantized inference serving |

> Results, plots, and analysis are added after benchmarking on a Colab T4 GPU.

---

## Repository Layout

```
triton-fused-kernels/
├── kernels/            # Triton kernel implementations
│   ├── rms_norm.py     #   Fused RMSNorm (two-pass, row-wise)
│   ├── swiglu.py       #   Fused SwiGLU (single-pass elementwise)
│   └── rms_norm_quant.py #  RMSNorm + INT8 quantization (two kernel launches)
├── baselines/          # Unfused PyTorch references (exact same numerics)
│   ├── rms_norm.py
│   └── swiglu.py
├── benchmark/
│   ├── config.py       # Sweep parameters, bandwidth formulas
│   ├── run.py          # Benchmark runner → results CSV
│   └── results/        # CSVs committed from Colab after each run
├── analysis/
│   ├── plot.py         # Generates figures from results CSVs (no GPU needed)
│   └── figures/        # PNG plots committed after local analysis
├── notebooks/
│   └── colab_run.ipynb # End-to-end Colab runner (setup → validate → bench → push)
└── requirements.txt
```

---

## Hardware

Benchmarks run on **NVIDIA Tesla T4** (Colab free tier):
- Architecture: Turing (Compute Capability 7.5)
- Memory: 16 GB GDDR6
- Peak memory bandwidth: **320 GB/s**
- Note: No native BF16 — kernels use FP32 (FP16 variant also supported)

---

## Kernel Algorithms

### RMSNorm
```
y = x / sqrt(mean(x²) + ε) * weight
```
- **Fused kernel**: one Triton program per token, two tile-loops over `hidden_size`
  - Pass 1: accumulate `sum(x²)` → compute `rms`
  - Pass 2: normalize and apply weight
- **Baseline**: `x.pow(2).mean(-1) → rsqrt → mul(weight)` (3 CUDA kernels)
- FP16 inputs are cast to FP32 inside the kernel for accumulation

### SwiGLU
```
y = silu(gate) * up    where silu(x) = x * sigmoid(x)
```
- **Fused kernel**: single pass — reads `gate` and `up`, writes `y` (3× memory traffic)
- **Baseline**: `gate * sigmoid(gate)` then `* up` (5× memory traffic — intermediate tensor)

### RMSNorm + INT8 Quantization
```
normalized = RMSNorm(x, weight)
scale = max(|normalized|, dim=token) / 127
quantized = round(clamp(normalized / scale, -128, 127)).to(int8)
```
- **Fused**: 2 Triton kernel launches, no persistent intermediate tensor
- **Baseline**: 4+ PyTorch CUDA kernels + intermediate float tensor

---

## Benchmark Methodology

- **Timing**: CUDA events per iteration — correct for async GPU execution
- **Warmup**: 100 runs before timing (triggers Triton JIT + autotuning)
- **Timed runs**: 200 iterations, reporting mean / median / std / p90
- **Bandwidth**: `bytes_accessed / (median_ms * 1e-3) / 1e9` using theoretical memory traffic
- **Sweep**: 5 hidden sizes × 6 (batch, seq_len) configs = 30 configurations per kernel

---

## Running on Colab

1. Open `notebooks/colab_run.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Fill in your GitHub username and email in Cell 4
4. Add a GitHub PAT to Colab Secrets (🔑 sidebar) as `GITHUB_PAT`
5. Run all 4 cells

Cell 1 (~3 min): installs dependencies and clones repo  
Cell 2 (~30 sec): validates kernel correctness  
Cell 3 (~5 min): runs full benchmark, saves CSV  
Cell 4 (~10 sec): commits and pushes results

> **Note**: Triton JIT compiles kernels on first call. Expect 30–60 seconds of compilation at the start of Cell 3. This is printed as progress — it is not hanging.

---

## Generating Plots Locally (no GPU needed)

After pulling the results CSVs from Colab:

```bash
git pull
python analysis/plot.py
```

Figures are saved to `analysis/figures/`:
- `speedup_vs_hidden_size.png` — Triton speedup over PyTorch per kernel
- `bandwidth_utilization.png` — GB/s vs T4 peak (roofline view)
- `heatmap_rms_norm.png` — latency heatmap
- `heatmap_swiglu.png`
- `heatmap_rms_norm_quant.png`

---

## Setup (local development, no GPU)

```bash
git clone https://github.com/shiva-sankeerth/triton-fused-kernels.git
cd triton-fused-kernels
pip install -r requirements.txt
```

PyTorch baseline code can be tested locally on CPU. Triton kernels require a CUDA GPU.

---

## Acknowledgements

Kernel designs inspired by:
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — production fused kernels for LLM training
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — tiling and online softmax patterns
- [OpenAI Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
