import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import sys

# ── Configuration ──
if len(sys.argv) > 1 and sys.argv[1] != "--all":
    ARITH_INTENSITY = int(sys.argv[1])
else:
    ARITH_INTENSITY = 256

CSV_INPUT_FILE = f"results_a{ARITH_INTENSITY}.csv"
PLOT_OUTPUT_FILE = f"synthetic_kernel_a{ARITH_INTENSITY}_h100.png"

# ── H100 parameters (4-byte loads, ÷128 normalization) ──
MEM_LAT    = 347
ALU_LAT    = 4
MEM_THRU   = 0.04230    # BW/(SMs × 128) — Volkov's 4-byte load normalization
ISSUE_THRU = 4.0        # H100 has 32 FP32 cores/partition → 4.0 IPC/SM
COEFF      = 6401.28   # 32 × 114 × 1.755


def volkov_model(n, ai):
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
    n_vals = np.linspace(0.1, max_occ * 1.05, 500)
    theo = volkov_model(n_vals, arith_intensity)

    ax.plot(n_vals, theo, color='black', linewidth=2.5, linestyle='--',
            label='Volkov Model', zorder=2)

    ax.plot(df["MaxAttainedOccupancy"], df["GFLOPS"],
            marker='o', markersize=10, linestyle='none',
            color='black', markerfacecolor='black',
            label='Measured', zorder=3)

    ax.set_xlabel("Warps / SM", fontsize=50, fontweight='bold', labelpad=10)
    ax.set_ylabel("GFLOP/s", fontsize=50, fontweight='bold', labelpad=10)
    ax.set_xlim(left=0, right=max_occ * 1.05)
    ax.set_ylim(bottom=0)

    ax.tick_params(axis='both', which='major', labelsize=36,
                   width=1.5, length=5, direction='in')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5,
            color='#aaaaaa', alpha=0.6)

    ax.legend(loc='lower right', frameon=False, fontsize=32, handlelength=2.0)

    plt.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.13)

    plt.savefig(output_path, bbox_inches="tight", dpi=350)
    plt.close()
    print(f"Plot saved: {output_path}")


def batch_generate():
    ai_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    generated = 0
    for ai in ai_values:
        csv_file = f"results_a{ai}.csv"
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file).sort_values("MaxAttainedOccupancy")
        if df.empty:
            continue
        out = f"synthetic_kernel_a{ai}_h100.png"
        generate_plot(df, ai, out)
        generated += 1
    print(f"\nGenerated {generated} plots.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        batch_generate()
    else:
        try:
            df = pd.read_csv(CSV_INPUT_FILE).sort_values("MaxAttainedOccupancy")
            if df.empty:
                print(f"❌ '{CSV_INPUT_FILE}' is empty.")
            else:
                generate_plot(df, ARITH_INTENSITY, PLOT_OUTPUT_FILE)
        except FileNotFoundError:
            print(f"❌ '{CSV_INPUT_FILE}' not found.")
        except KeyError as e:
            print(f"❌ Missing column: {e}")
