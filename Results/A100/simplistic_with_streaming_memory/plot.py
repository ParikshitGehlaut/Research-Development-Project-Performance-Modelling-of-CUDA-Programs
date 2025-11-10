import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# --- Configuration ---
ARITH_INTENSITY = 1  # α = arithmetic intensity
CSV_INPUT_FILE = f"results_a{ARITH_INTENSITY}.csv"
PLOT_OUTPUT_FILE = f"simplistic_kernel_a{ARITH_INTENSITY}_a100.png"

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
    Arithmetic Throughput (GFLOPS) = 6289.92 * α * min( n / (265 + α*4), 0.0406, 4 / (α + 1) )
    """
    # 32 * 108 * 1.410 = 4872.96
    coeff = 4872.96 * arith_intensity
    term1 = warps_per_sm / (194 + arith_intensity * 4)
    term2 = 0.0421
    term3 = 2 / (arith_intensity + 1)
    
    # Using np.full_like to ensure 'term2' and 'term3' are arrays of the same shape as 'warps_per_sm'
    bottleneck = np.minimum.reduce([
        term1, 
        np.full_like(warps_per_sm, term2), 
        np.full_like(warps_per_sm, term3)
    ])
    return coeff * bottleneck

def generate_plot(df, arith_intensity, output_path):
    """Generates and saves the performance profile plot."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 1. Plot Experimental Data (as a scatter plot)
    ax.plot(
        df["MaxAttainedOccupancy"], 
        df["GFLOPS"], 
        marker="o", 
        markersize=5, 
        linestyle="none", 
        color="black", 
        label="Experimental Data"
    )
    
    # 2. Plot Theoretical Model
    #    Ensure min/max are taken *after* sorting
    min_occupancy = df["MaxAttainedOccupancy"].min()
    max_occupancy = df["MaxAttainedOccupancy"].max()
    
    n_vals = np.linspace(0, max_occupancy, 200) # Start from 0 for the model
    theoretical_gflops = calculate_simplistic_kernel_model(n_vals, arith_intensity)
    
    ax.plot(
        n_vals, 
        theoretical_gflops, 
        linestyle="--", 
        color="dimgray", 
        label=f"Theoretical (α={arith_intensity})"
    )

    # --- Labels and Formatting ---
    ax.set_xlabel("Warps / SM")
    ax.set_ylabel("GFLOP/s")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0) # Start x-axis at 0
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc="best", frameon=False)
    
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Successfully generated plot: '{output_path}'")

if __name__ == "__main__":
    try:
        # Load the data
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        
        # Sort values to ensure min/max for linspace are correct
        dataframe = dataframe.sort_values("MaxAttainedOccupancy")
        
        if dataframe.empty:
            print(f"❌ Error: The file '{CSV_INPUT_FILE}' is empty.")
        else:
            generate_plot(dataframe, ARITH_INTENSITY, PLOT_OUTPUT_FILE)
        
    except FileNotFoundError:
        print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")
        print("Please make sure your data is saved in this file.")
    except KeyError as e:
        print(f"❌ Error: A required column is missing from the CSV.")
        print(f"Could not find column: {e}")
