import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
ARITH_INTENSITY = 1  # α = arithmetic intensity
CSV_INPUT_FILE = f"results_a{ARITH_INTENSITY}.csv"
PLOT_OUTPUT_FILE = f"simplistic_kernel_a{ARITH_INTENSITY}_rtx.png"

def setup_plot_style():
    """Sets a bold, high-visibility plot style suitable for small subfigures."""
    plt.rcParams.update({
        # significantly increased font sizes
        'font.size': 18,
        'font.family': 'serif',
        'axes.labelsize': 22,      # X and Y axis labels
        'axes.titlesize': 22,
        'legend.fontsize': 16,     # Legend text
        'xtick.labelsize': 18,     # Tick numbers
        'ytick.labelsize': 18,
        
        # Thicker lines and axes
        'lines.linewidth': 4,      # Thicker theoretical line
        'axes.linewidth': 2,       # Thicker border
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })

def calculate_simplistic_kernel_model(warps_per_sm, arith_intensity):
    # 32 * 48 * 1.815 = 2787.84
    coeff = 2787.84 * arith_intensity
    term1 = warps_per_sm / (431 + arith_intensity * 4)
    term2 = 0.000715  # mem_thru: corrected for 8-byte loads (÷256 instead of ÷128)
    term3 = 4 / (arith_intensity + 1)
    
    bottleneck = np.minimum.reduce([
        term1, 
        np.full_like(warps_per_sm, term2), 
        np.full_like(warps_per_sm, term3)
    ])
    return coeff * bottleneck

def generate_plot(df, arith_intensity, output_path):
    """Generates and saves the performance profile plot."""
    setup_plot_style()
    
    # Kept figsize small (6x4) so that relative font size appears larger
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 1. Plot Experimental Data (scatter)
    ax.plot(
        df["MaxAttainedOccupancy"], 
        df["GFLOPS"], 
        marker="o", 
        markersize=12,       # Increased from 5 to 12
        linestyle="none", 
        color="black", 
        label="Experimental" # Shortened label for cleaner legend
    )
    
    # 2. Plot Theoretical Model
    min_occupancy = df["MaxAttainedOccupancy"].min()
    max_occupancy = df["MaxAttainedOccupancy"].max()
    
    n_vals = np.linspace(0, max_occupancy, 200)
    theoretical_gflops = calculate_simplistic_kernel_model(n_vals, arith_intensity)
    
    ax.plot(
        n_vals, 
        theoretical_gflops, 
        linestyle="--", 
        color="dimgray", 
        linewidth=3.5,       # Explicitly set thick line
        label=f"Theoretical (α={arith_intensity})"
    )

    # --- Labels and Formatting ---
    ax.set_xlabel("Warps / SM", fontweight='bold') # Bold label
    ax.set_ylabel("GFLOP/s", fontweight='bold')    # Bold label
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    # Thicker, darker grid
    ax.grid(True, which='major', linestyle='--', linewidth=1.5, color='gray', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend location
    ax.legend(loc="best", frameon=False)
    
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Successfully generated high-visibility plot: '{output_path}'")

if __name__ == "__main__":
    try:
        # Load the data
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        dataframe = dataframe.sort_values("MaxAttainedOccupancy")
        
        if dataframe.empty:
            print(f"❌ Error: The file '{CSV_INPUT_FILE}' is empty.")
        else:
            generate_plot(dataframe, ARITH_INTENSITY, PLOT_OUTPUT_FILE)
        
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")
    except KeyError as e:
        print(f"❌ Error: Missing column: {e}")