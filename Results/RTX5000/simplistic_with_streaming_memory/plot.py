import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
ARITH_INTENSITY = 512  # α = arithmetic intensity

CSV_INPUT_FILE = f"results_a{ARITH_INTENSITY}.csv"
PLOT_OUTPUT_FILE = f"simplistic_kernel_a{ARITH_INTENSITY}_rtx.png"

def setup_plot_style():
    """Sets a professional, publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 7,
        'axes.linewidth': 1.5,
    })

def calculate_simplistic_kernel_model(warps_per_sm, arith_intensity):
    """
    Theoretical arithmetic throughput model for the Simplistic Kernel.
    
    Arithmetic Throughput (GFLOPS) =
        2787.84 * α * min( n / (424 + α*4.38), 0.04017, 4 / (α + 1) )
    """
    coeff = 2787.84 * arith_intensity
    
    term1 = warps_per_sm / (424 + arith_intensity * 4.38)
    term2 = 0.04017
    term3 = 4 / (arith_intensity + 1)
    
    bottleneck = np.minimum.reduce([term1, np.full_like(warps_per_sm, term2), np.full_like(warps_per_sm, term3)])
    
    return coeff * bottleneck

def generate_plot(df, arith_intensity, output_path):
    """Generates and saves the performance profile plot for the simplistic kernel."""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

    # 1. Plot Experimental Data
    ax.plot(df["MaxAttainedOccupancy"], df["GFLOPS"],
            marker="o", linestyle="-", color="black",
            label="Experimental")

    # 2. Plot Theoretical Model
    n_vals = np.linspace(df["MaxAttainedOccupancy"].min(),
                         df["MaxAttainedOccupancy"].max(), 200)
    theoretical_gflops = calculate_simplistic_kernel_model(n_vals, arith_intensity)
    
    ax.plot(n_vals, theoretical_gflops, linestyle="--", color="dimgray", 
            label=f"Theoretical (α={arith_intensity})")

    # --- Labels and Formatting ---
    ax.set_xlabel("Warps / SM")
    ax.set_ylabel("GFLOP/s")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=df["MaxAttainedOccupancy"].min() * 0.95)
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="best", frameon=False)
    
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Successfully generated Simplistic Kernel plot: '{output_path}'")

if __name__ == "__main__":
    try:
        # Load and sort the data
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        dataframe = dataframe.sort_values("MaxAttainedOccupancy")
        generate_plot(dataframe, ARITH_INTENSITY, PLOT_OUTPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")
