import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ── Configuration ──
if len(sys.argv) > 1 and sys.argv[1] != "--all":
    ARITH_INTENSITY = int(sys.argv[1])
else:
    ARITH_INTENSITY = 256

CSV_INPUT_FILE = f"results_a{ARITH_INTENSITY}.csv"
PLOT_OUTPUT_FILE = f"simplistic_kernel_a{ARITH_INTENSITY}_a100.png"

# ── A100 parameters ──
MEM_LAT    = 240
ALU_LAT    = 4
MEM_THRU   = 0.0324     # corrected for 8-byte loads (÷256)
ISSUE_THRU = 2.0
COEFF      = 4872.96    # 32 × 108 × 1.410


def volkov_model(n, ai):
    latency_bound = n / (MEM_LAT + ai * ALU_LAT)
    memory_bound  = np.full_like(n, MEM_THRU)
    issue_bound   = np.full_like(n, ISSUE_THRU / (ai + 1))
    T_m = np.minimum.reduce([latency_bound, memory_bound, issue_bound])
    return COEFF * ai * T_m


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })


def generate_plot(df, arith_intensity, output_path):
    setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    max_occ = df["MaxAttainedOccupancy"].max()
    n_vals = np.linspace(0.1, max_occ * 1.05, 500)
    theo = volkov_model(n_vals, arith_intensity)

    # Theoretical model — solid dark line
    ax.plot(n_vals, theo, color='black', linewidth=2.0, linestyle='--',
            label='Theoretical', zorder=2)

    # Experimental data — filled circles
    ax.plot(df["MaxAttainedOccupancy"], df["GFLOPS"],
            marker='o', markersize=7, linestyle='none',
            color='black', markerfacecolor='black',
            label='Experimental', zorder=3)

    # ── Formatting ──
    ax.set_xlabel("Warps / SM", fontweight='bold')
    ax.set_ylabel("GFLOP/s", fontweight='bold')
    ax.set_xlim(left=0, right=max_occ * 1.05)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, which='major', linestyle='-', linewidth=0.4,
            color='#cccccc', alpha=0.7)

    ax.legend(loc='best', frameon=False)

    fig.tight_layout(pad=0.5)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Plot saved: {output_path}")


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
        out = f"simplistic_kernel_a{ai}_a100.png"
        generate_plot(df, ai, out)
        generated += 1
    print(f"\n✅ Generated {generated} plots.")


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