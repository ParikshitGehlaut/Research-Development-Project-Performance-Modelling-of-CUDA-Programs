import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import sys

# ── RTX 5000 Verified Constants (4-byte loads, Volkov's Methodology) ──
MEM_LAT    = 236        # Unloaded memory latency (cycles)
ALU_LAT    = 4          # FMUL latency (cycles)
MEM_THRU   = 0.0305     # 340 GB/s / (1.815 GHz × 48 SMs × 128 B/warp-instr)
ISSUE_THRU = 2.0        # Warp-instructions per cycle per SM
COEFF      = 2787.84    # 32 threads/warp × 48 SMs × 1.815 GHz


def volkov_model(n, ai):
    """Volkov's throughput model: P = C × AI × min(latency, memory, issue)"""
    n = np.asarray(n, dtype=float)
    latency_bound = n / (MEM_LAT + ai * ALU_LAT)
    memory_bound  = np.full_like(n, MEM_THRU)
    issue_bound   = np.full_like(n, ISSUE_THRU / (ai + 1))
    T_m = np.minimum.reduce([latency_bound, memory_bound, issue_bound])
    return COEFF * ai * T_m


def setup_style():
    """Minimal global style — fonts only; sizes set per-axis for precision."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
    })


def generate_plot(df, arith_intensity, output_path):
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=350)

    max_occ = df["MaxAttainedOccupancy"].max()
    n_vals = np.linspace(0.1, max_occ * 1.1, 500)
    theo = volkov_model(n_vals, arith_intensity)

    ax.plot(n_vals, theo, color='black', linewidth=2.5, linestyle='--',
            label='Volkov Model', zorder=2)

    ax.scatter(df["MaxAttainedOccupancy"], df["GFLOPS"],
               marker='o', s=144, color='black', facecolors='black',
               label='Measured', zorder=3)

    ax.set_xlabel("Warps / SM", fontsize=64, fontweight='bold', labelpad=10)
    ax.set_ylabel("GFLOP/s", fontsize=64, fontweight='bold', labelpad=10)
    ax.set_xlim(left=0, right=max_occ * 1.1)
    ax.set_ylim(bottom=0)

    ax.tick_params(axis='both', which='major', labelsize=54,
                   width=1.5, length=5, direction='in')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5,
            color='#aaaaaa', alpha=0.6)

    ax.legend(loc='lower right', frameon=False, fontsize=48, handlelength=2.0)

    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.13)

    plt.savefig(output_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"Saved: {output_path}")


def batch_generate():
    ai_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    generated = 0
    for ai in ai_values:
        csv_file = f"results_a{ai}.csv"
        if not os.path.exists(csv_file):
            print(f"⚠️  Skipping AI={ai}: {csv_file} not found")
            continue
        df = pd.read_csv(csv_file).sort_values("MaxAttainedOccupancy")
        if df.empty:
            continue
        out = f"synthetic_kernel_a{ai}_rtx5000.png"
        generate_plot(df, ai, out)
        generated += 1
    print(f"\nGenerated {generated} plots for RTX 5000.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        batch_generate()
    elif len(sys.argv) > 1:
        ai = int(sys.argv[1])
        csv_file = f"results_a{ai}.csv"
        try:
            df = pd.read_csv(csv_file).sort_values("MaxAttainedOccupancy")
            generate_plot(df, ai, f"synthetic_kernel_a{ai}_rtx5000.png")
        except FileNotFoundError:
            print(f"❌ '{csv_file}' not found.")
    else:
        print("Usage: python plot.py <AI_value> | python plot.py --all")
