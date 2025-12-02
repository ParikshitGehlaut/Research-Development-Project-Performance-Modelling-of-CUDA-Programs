import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
ARITH_INTENSITY = 0.5  # Fixed for naive matmul (alpha)
CSV_INPUT_FILE = "results_n4096.csv"
PLOT_OUTPUT_FILE = "naive_rtx5000.png"

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

def calculate_volkov_model(warps_per_sm, alpha):
    """
    Calculates Arithmetic Throughput (GFLOPS) based on Volkov's formula:
    Throughput = 32 * alpha * NumSMs * Clock * min(n/Latency, mem_thru, issue_thru/(alpha+1)) * 2
    """
    
    # --- Hardware Constants (H100 inferred) ---
    # num_sms (114) * clock (1.755) * 32 ~= 6402.24 (base coeff from previous script)
    num_sms = 48           
    clock_rate_ghz = 1.815  
    warp_size = 32
    
    # --- Constraints ---
    latency = 431        # L (cycles)
    mem_thru = 0.0315       # Memory ops per cycle per SM
    issue_thru = 4.0        # Issue bandwidth (instructions per cycle per SM)
    
    # 1. Calculate the Memory Throughput Term: min( n/L, mem_thru, issue/(a+1) )
    #    'n' is warps_per_sm
    term_latency = warps_per_sm / (latency + 4 * alpha)
    term_mem_bw  = np.full_like(warps_per_sm, mem_thru)
    term_issue   = np.full_like(warps_per_sm, issue_thru / (alpha + 1))
    
    min_throughput_term = np.minimum.reduce([
        term_latency, 
        term_mem_bw, 
        term_issue
    ])
    
    # 2. Scale to Arithmetic Throughput (GFLOPS)
    #    Formula: 32 * alpha * SMs * Clock * min(...) * 2 (for FMA)
    gflops = (warp_size * alpha * num_sms * clock_rate_ghz) * min_throughput_term * 2
    
    return gflops

def generate_plot(df, arith_intensity, output_path):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 1. Plot Experimental Data (Black Scatter Dots)
    ax.plot(
        df["MaxAttainedOccupancy"], 
        df["GFLOPS"], 
        marker="o", 
        markersize=5, 
        linestyle="none", 
        color="black", 
        label="Experimental",
        alpha=0.75
    )
    
    # 2. Plot Theoretical Model (Volkov)
    max_occupancy = df["MaxAttainedOccupancy"].max()
    # Generate X-axis values (Warps per SM)
    n_vals = np.linspace(0, max_occupancy * 1.1, 300) 
    
    theoretical_gflops = calculate_volkov_model(n_vals, arith_intensity)
    
    ax.plot(
        n_vals, 
        theoretical_gflops, 
        linestyle="--", 
        color="dimgray", 
        linewidth=2,
        label=f"Theoretical ($\\alpha={arith_intensity}$)"
    )

    # --- Labels and Formatting ---
    ax.set_xlabel("Warps / SM (Occupancy)")
    ax.set_ylabel("Arithmetic Throughput (GFLOP/s)")
    ax.set_title("Performance vs Occupancy (Naive MatMul)")
    
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=max_occupancy * 1.1)
    
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc="lower right", frameon=False)
    
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ Successfully generated plot: '{output_path}'")

if __name__ == "__main__":
    try:
        if not os.path.exists(CSV_INPUT_FILE):
            print(f"❌ Error: The file '{CSV_INPUT_FILE}' was not found.")
            exit(1)

        # Load the data
        dataframe = pd.read_csv(CSV_INPUT_FILE)
        dataframe.columns = dataframe.columns.str.strip()

        if dataframe.empty:
            print(f"❌ Error: The file '{CSV_INPUT_FILE}' is empty.")
        else:
            dataframe = dataframe.sort_values("MaxAttainedOccupancy")
            generate_plot(dataframe, ARITH_INTENSITY, PLOT_OUTPUT_FILE)
        
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")