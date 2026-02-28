import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CSV_INPUT_FILE = "results.csv"
PLOT_OUTPUT_FILE = "naive_h100.png"

def setup_plot_style():
    """Sets a professional, publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,  # Increased base font size
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 14,   # CHANGED: Increased from 11 for better visibility
        'legend.fontsize': 12,  # CHANGED: Increased from 9 for better legibility
        'xtick.labelsize': 12,  # CHANGED: Increased from 9 to make ticks clearer
        'ytick.labelsize': 12,  # CHANGED: Increased from 9 to make ticks clearer
        'lines.linewidth': 2.5, # Slightly thicker lines
        'lines.markersize': 7,  # Slightly larger markers
        'axes.linewidth': 1.5
    })

def calculate_theoretical_model(warps_per_sm):
    """
    Calculates the theoretical performance based on a GPU model.
    The model is bounded by latency, compute, and bandwidth limitations.
    """
    # --- Model Parameters ---
    T = 1
    PEAK_PERF_CONSTANT = 12804.48 * T
    LATENCY = 660 + (T * 4)
    BANDWIDTH_LIMIT = 0.0781
    COMPUTE_LIMIT = 4 / (T + 1)

    # Performance is the minimum of the three limiters, scaled by the peak constant.
    latency_bound = warps_per_sm / LATENCY
    
    bottleneck = np.minimum.reduce([
        latency_bound,
        np.full_like(warps_per_sm, BANDWIDTH_LIMIT),
        np.full_like(warps_per_sm, COMPUTE_LIMIT)
    ])
    
    return PEAK_PERF_CONSTANT * bottleneck

def generate_plot(df, output_path):
    """Generates and saves the GPU performance profile plot."""
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # 1. Plot Experimental Data
    ax.plot(df["MaxAttainedOccupancy"], df["GFLOPS"],
            marker="o", linestyle="-", color="black",
            label="Experimental")

    # 2. Plot Theoretical Model
    n_vals = np.linspace(df["MaxAttainedOccupancy"].min(),
                         df["MaxAttainedOccupancy"].max(), 200)
    theoretical_gflops = calculate_theoretical_model(n_vals)
    
    ax.plot(n_vals, theoretical_gflops, linestyle="--", color="dimgray", 
            label="Theoretical")

    # --- Labels and Formatting ---
    ax.set_xlabel("Warps / SM")
    ax.set_ylabel("Gflop/s")
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=df["MaxAttainedOccupancy"].min() * 0.95)

    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="best", frameon=False)
    
    fig.tight_layout()
    
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Successfully generated publication-quality plot: '{output_path}'")


if __name__ == "__main__":
    # Load and sort the data by occupancy
    try:
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        dataframe = dataframe.sort_values("MaxAttainedOccupancy")
        generate_plot(dataframe, PLOT_OUTPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")