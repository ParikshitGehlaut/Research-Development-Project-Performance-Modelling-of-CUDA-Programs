import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CSV_INPUT_FILE = "results_H100_tile_dim_16.csv"
PLOT_OUTPUT_FILE = "t16_h100.png"
T_VALUE = 16

def setup_plot_style():
    """Sets a professional, publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,  # Increased base font size for overall legibility
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 14,   # CHANGED: Increased from 11 for clear axis titles
        'legend.fontsize': 12,  # CHANGED: Increased from 9 for a readable legend
        'xtick.labelsize': 12,  # CHANGED: Increased from 9 for visible tick numbers
        'ytick.labelsize': 12,  # CHANGED: Increased from 9 for visible tick numbers
        'lines.linewidth': 2.5, # CHANGED: Thicker lines to make data stand out
        'lines.markersize': 7,  # CHANGED: Larger markers for better visibility
        'axes.linewidth': 1.5,
    })

def calculate_theoretical_model(warps_per_sm, T):
    """
    Calculates the theoretical performance based on a GPU model.
    The model is bounded by latency, compute, and bandwidth limitations.
    """
    # --- Model Parameters ---
    PEAK_PERF_CONSTANT = 12804.48
    LATENCY_BASE = 500
    LATENCY_SCALING_FACTOR = 4
    BANDWIDTH_LIMIT = 0.0781
    COMPUTE_LIMIT_NUMERATOR = 4

    # Calculate the performance bounds
    latency = LATENCY_BASE + (T * LATENCY_SCALING_FACTOR)
    latency_bound = warps_per_sm / latency
    compute_bound = COMPUTE_LIMIT_NUMERATOR / (T + 1)
    
    # Performance is the minimum of the three limiters, scaled by constants
    bottleneck = np.minimum.reduce([
        latency_bound,
        np.full_like(warps_per_sm, BANDWIDTH_LIMIT),
        np.full_like(warps_per_sm, compute_bound)
    ])
    
    return (PEAK_PERF_CONSTANT * T) * bottleneck

def generate_plot(df, T, output_path):
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
    theoretical_gflops = calculate_theoretical_model(n_vals, T)
    
    ax.plot(n_vals, theoretical_gflops, linestyle="--", color="dimgray", 
            label=f"Theoretical (T={T})")

    # --- Labels and Formatting ---
    # CHANGED: Using more concise labels for a cleaner look
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
    try:
        # Load and sort the data by occupancy
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        dataframe = dataframe.sort_values("MaxAttainedOccupancy")
        generate_plot(dataframe, T_VALUE, PLOT_OUTPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")