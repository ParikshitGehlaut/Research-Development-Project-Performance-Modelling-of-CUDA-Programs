import pandas as pd
import matplotlib.pyplot as plt
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
    n_vals = np.linspace(0.1, max_occ * 1.1, 500)
    theo = volkov_model(n_vals, arith_intensity)

    # Theoretical curve
    ax.plot(n_vals, theo, color='black', linewidth=2.0, linestyle='--',
            label='Volkov Model', zorder=2)

    # Experimental data
    ax.scatter(df["MaxAttainedOccupancy"], df["GFLOPS"],
               marker='o', s=50, color='black', facecolors='black',
               label='Measured', zorder=3)

    ax.set_xlabel("Occupancy (warps/SM)", fontweight='bold')
    ax.set_ylabel("GFLOP/s", fontweight='bold')
    ax.set_title(f"AI = {arith_intensity}", fontsize=14, fontweight='bold')
    ax.set_xlim(left=0, right=max_occ * 1.1)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(True, which='major', linestyle='-', linewidth=0.4,
            color='#cccccc', alpha=0.7)

    ax.legend(loc='best', frameon=False)

    fig.tight_layout(pad=0.5)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}")


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
        out = f"volkov_rtx5000_a{ai}.png"
        generate_plot(df, ai, out)
        generated += 1
    print(f"\n✅ Generated {generated} plots for RTX 5000.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        batch_generate()
    elif len(sys.argv) > 1:
        ai = int(sys.argv[1])
        csv_file = f"results_a{ai}.csv"
        try:
            df = pd.read_csv(csv_file).sort_values("MaxAttainedOccupancy")
            generate_plot(df, ai, f"volkov_rtx5000_a{ai}.png")
        except FileNotFoundError:
            print(f"❌ '{csv_file}' not found.")
    else:
        print("Usage: python plot.py <AI_value> | python plot.py --all")
