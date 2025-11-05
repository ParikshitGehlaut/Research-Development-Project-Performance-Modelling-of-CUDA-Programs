import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CSV_FILE = 'Results/RTX5000/Latency/latency_summary.csv'
OUTPUT_IMAGE = 'Results/RTX5000/Latency/latency_heatmap.png'
PLOT_TITLE = 'GPU Memory Latency (Cycles) vs. Occupancy Controls (RTX 5000)'
# --- End Configuration ---

def create_plot():
    print(f"Reading data from {CSV_FILE}...")
    if not os.path.exists(CSV_FILE):
        print(f"Error: File not found at {CSV_FILE}")
        print("Please run the ./run_memory_latency.sh script first.")
        return

    # Read the CSV data
    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        print(f"Error: {CSV_FILE} is empty. No data to plot.")
        return

    # Clean the data:
    # 1. Remove any 'FAIL' rows
    df = df[df['MinLatency_cycles'].astype(str).str.lower() != 'fail']
    # 2. Convert columns to numeric types
    df['ThreadsPerBlock'] = pd.to_numeric(df['ThreadsPerBlock'])
    df['Shmem_KB'] = pd.to_numeric(df['Shmem_KB'])
    df['MinLatency_cycles'] = pd.to_numeric(df['MinLatency_cycles'])

    if df.empty:
        print("Error: No valid data found after cleaning.")
        return

    # Find the minimum latency
    min_lat = df['MinLatency_cycles'].min()
    print(f"Minimum (unloaded) latency found in data: {min_lat:.2f} cycles")

    # Pivot the data into a 2D grid for the heatmap
    # 'ThreadsPerBlock' will be the rows (y-axis)
    # 'Shmem_KB' will be the columns (x-axis)
    # 'MinLatency_cycles' will be the cell values
    try:
        pivot_table = df.pivot(
            index='ThreadsPerBlock', 
            columns='Shmem_KB', 
            values='MinLatency_cycles'
        )
    except ValueError as e:
        print(f"Error pivoting data: {e}")
        print("This can happen if you have duplicate (TPB, Shmem_KB) entries.")
        return

    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Draw the heatmap
    # annot=True writes the latency value in each cell
    # fmt='.1f' formats the value to one decimal place
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".1f", 
        cmap="viridis_r",  # '_r' reverses the colormap (blue=low, yellow=high)
        linewidths=.5,
        cbar_kws={'label': 'Minimum Latency (cycles)'} # Color bar label
    )
    
    # Set titles and labels
    ax.set_title(PLOT_TITLE, fontsize=16, pad=20)
    ax.set_xlabel("Shared Memory per Block (KB)", fontsize=12)
    ax.set_ylabel("Threads Per Block", fontsize=12)
    
    # Invert y-axis to have smaller TPB at the bottom (optional, but common)
    # ax.invert_yaxis() 

    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    
    print(f"\nSuccess! Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    create_plot()