"""
Generate benchmark plots from CSV results.

Usage:
    python analysis/plot.py [--results-dir benchmark/results] [--output-dir analysis/figures]

Outputs (in analysis/figures/):
    speedup_vs_hidden_size.png     — Triton speedup over PyTorch per kernel
    bandwidth_utilization.png      — Memory bandwidth (GB/s) vs T4 peak
    heatmap_rms_norm.png           — Latency heatmap for RMSNorm
    heatmap_swiglu.png             — Latency heatmap for SwiGLU
    heatmap_rms_norm_quant.png     — Latency heatmap for RMSNorm+Quant

No GPU required — runs locally on Mac.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
T4_PEAK_BW_GBS = 320.0

KERNEL_LABELS = {
    "rms_norm":       "RMSNorm",
    "swiglu":         "SwiGLU",
    "rms_norm_quant": "RMSNorm+INT8",
}

COLORS = {
    "rms_norm":       "#4C72B0",
    "swiglu":         "#DD8452",
    "rms_norm_quant": "#55A868",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

def load_results(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("results_*.csv"))
    if not csvs:
        print(f"No result CSVs found in {results_dir}")
        print("Run the benchmark on Colab first, then pull the results.")
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

    # Keep most recent run per unique config
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["kernel_name", "variant", "hidden_size", "num_tokens", "dtype"],
            keep="last",
        )
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} rows from {len(csvs)} CSV(s)")
    return df


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    triton_df  = df[df["variant"] == "triton"].copy()
    pytorch_df = df[df["variant"] == "pytorch"].copy()

    merged = triton_df.merge(
        pytorch_df,
        on=["kernel_name", "hidden_size", "num_tokens", "dtype"],
        suffixes=("_triton", "_pytorch"),
    )
    merged["speedup"] = merged["median_ms_pytorch"] / merged["median_ms_triton"]
    return merged


# ------------------------------------------------------------------ #
# Plot 1 — Speedup vs hidden_size                                     #
# ------------------------------------------------------------------ #

def plot_speedup(speedup_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for kernel_name, label in KERNEL_LABELS.items():
        kdf = speedup_df[speedup_df["kernel_name"] == kernel_name]
        if kdf.empty:
            continue
        # Average speedup across token configs for each hidden_size
        grouped = kdf.groupby("hidden_size")["speedup"].agg(["mean", "std"]).reset_index()
        ax.plot(
            grouped["hidden_size"], grouped["mean"],
            marker="o", label=label, color=COLORS[kernel_name], linewidth=2,
        )
        ax.fill_between(
            grouped["hidden_size"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15, color=COLORS[kernel_name],
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Break-even (1×)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xlabel("Hidden Size")
    ax.set_ylabel("Speedup (PyTorch time / Triton time)")
    ax.set_title("Triton Speedup over PyTorch Baseline")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / "speedup_vs_hidden_size.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------ #
# Plot 2 — Memory bandwidth utilization                               #
# ------------------------------------------------------------------ #

def plot_bandwidth(df: pd.DataFrame, output_dir: Path):
    kernels = [k for k in KERNEL_LABELS if df[df["kernel_name"] == k].shape[0] > 0]
    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 5), sharey=False)
    if len(kernels) == 1:
        axes = [axes]

    for ax, kernel_name in zip(axes, kernels):
        kdf = df[df["kernel_name"] == kernel_name]

        for variant, ls, marker in [("triton", "-", "o"), ("pytorch", "--", "s")]:
            vdf = kdf[kdf["variant"] == variant]
            if vdf.empty:
                continue
            grouped = vdf.groupby("hidden_size")["bandwidth_gbs"].mean().reset_index()
            label = f"Triton" if variant == "triton" else "PyTorch"
            ax.plot(
                grouped["hidden_size"], grouped["bandwidth_gbs"],
                linestyle=ls, marker=marker, label=label,
                color=COLORS[kernel_name], linewidth=2,
                alpha=1.0 if variant == "triton" else 0.6,
            )

        ax.axhline(
            T4_PEAK_BW_GBS, color="red", linestyle=":", linewidth=1.5,
            label=f"T4 Peak ({T4_PEAK_BW_GBS} GB/s)",
        )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xlabel("Hidden Size")
        ax.set_ylabel("Memory Bandwidth (GB/s)")
        ax.set_title(KERNEL_LABELS[kernel_name])
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Memory Bandwidth Utilization vs T4 Theoretical Peak", fontsize=13, y=1.01)
    path = output_dir / "bandwidth_utilization.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------ #
# Plot 3 — Latency heatmaps                                          #
# ------------------------------------------------------------------ #

def plot_heatmap(df: pd.DataFrame, kernel_name: str, output_dir: Path):
    kdf = df[df["kernel_name"] == kernel_name].copy()
    if kdf.empty:
        return

    kdf["token_config"] = kdf.apply(
        lambda r: f"{r['batch_size']}×{r['seq_len']}", axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vmin = kdf["median_ms"].min()
    vmax = kdf["median_ms"].max()

    for ax, variant in zip(axes, ["triton", "pytorch"]):
        vdf = kdf[kdf["variant"] == variant]
        if vdf.empty:
            ax.set_visible(False)
            continue

        pivot = vdf.pivot_table(
            index="token_config", columns="hidden_size",
            values="median_ms", aggfunc="mean",
        )
        # Sort rows by total tokens
        pivot = pivot.reindex(
            sorted(pivot.index, key=lambda s: int(s.split("×")[0]) * int(s.split("×")[1]))
        )

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
            cbar_kws={"label": "Median latency (ms)"},
            linewidths=0.5,
        )
        ax.set_title(f"{KERNEL_LABELS[kernel_name]} — {variant.capitalize()}")
        ax.set_xlabel("Hidden Size")
        ax.set_ylabel("(batch × seq_len)")

    fig.suptitle(f"Latency Heatmap: {KERNEL_LABELS[kernel_name]}", fontsize=13)
    path = output_dir / f"heatmap_{kernel_name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--results-dir", default="benchmark/results")
    parser.add_argument("--output-dir", default="analysis/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    speedup_df = compute_speedups(df)

    print("\n--- Headline speedups (mean across configs) ---")
    for kernel_name in KERNEL_LABELS:
        kdf = speedup_df[speedup_df["kernel_name"] == kernel_name]
        if not kdf.empty:
            mean_sp = kdf["speedup"].mean()
            max_sp  = kdf["speedup"].max()
            print(f"  {KERNEL_LABELS[kernel_name]:20s}  mean {mean_sp:.2f}×  max {max_sp:.2f}×")

    plot_speedup(speedup_df, output_dir)
    plot_bandwidth(df, output_dir)
    for kernel_name in KERNEL_LABELS:
        plot_heatmap(df, kernel_name, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
