import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Configuration ---
# You can run this script like: python plot_tiled.py Results/tiled/results_tiled_n4096_t16.csv
DEFAULT_INPUT_FILE = "results_tiled_n4096_t16.csv"

def setup_plot_style():
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif',
        'axes.labelsize': 14, 'legend.fontsize': 12,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'lines.linewidth': 2.5, 'lines.markersize': 7,
        'axes.linewidth': 1.5,
    })

def calculate_volkov_model(warps_per_sm, alpha):
    """
    Volkov Model: GFLOPS ~ min(Latency, Bandwidth, Issue)
    Alpha = T / 2 for Tiled MatMul
    Formula mult by 2 because 1 FMA = 2 FLOPs
    """
    # H100/A100 estimates (Adjust if needed for specific GPU)
    num_sms = 108           # A100=108, H100=114
    clock_rate_ghz = 1.410   # A100 Boost
    warp_size = 32
    
    # Roofline constraints
    latency = 240.0         # Latency L (cycles)
    mem_thru = 0.0648       # Memory ops per cycle per SM (approx)
    issue_thru = 2.0        # Issue bandwidth
    
    # Terms
    term_latency = warps_per_sm / (latency + 4 * alpha)
    term_mem     = np.full_like(warps_per_sm, mem_thru)
    term_issue   = np.full_like(warps_per_sm, issue_thru / (alpha + 1))
    
    bottleneck = np.minimum.reduce([term_latency, term_mem, term_issue])
    
    # Throughput in Instructions/sec * 2 FLOPs/Instr
    gflops = (warp_size * alpha * num_sms * clock_rate_ghz) * bottleneck * 2
    return gflops

def generate_plot(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Determine Tile Size from data or filename
    if 'TILE_DIM' in df.columns:
        tile_dim = df['TILE_DIM'].iloc[0]
    else:
        # Fallback extraction
        import re
        match = re.search(r'_t(\d+)', csv_path)
        tile_dim = int(match.group(1)) if match else 16
    
    # Arithmetic Intensity for Tiled MatMul: Alpha = T / 2
    alpha = tile_dim / 2.0
    
    output_file = csv_path.replace(".csv", ".png")
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 1. Experimental Data
    df = df.sort_values("MaxAttainedOccupancy")
    ax.plot(
        df["MaxAttainedOccupancy"], 
        df["GFLOPS"], 
        marker="o", markersize=6, linestyle="none", 
        color="black", alpha=0.8, label=f"Experimental (T={tile_dim})"
    )

    # 2. Theoretical Model
    max_occ = df["MaxAttainedOccupancy"].max()
    n_vals = np.linspace(0, max_occ * 1.2, 300)
    
    model_y = calculate_volkov_model(n_vals, alpha)
    
    ax.plot(
        n_vals, model_y, 
        linestyle="--", color="dimgray", 
        label=f"Theoretical ($\\alpha={alpha}$)"
    )

    ax.set_xlabel("Warps / SM")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Tiled MatMul Performance (T={tile_dim})")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"✅ Plot saved to {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    generate_plot(input_file)