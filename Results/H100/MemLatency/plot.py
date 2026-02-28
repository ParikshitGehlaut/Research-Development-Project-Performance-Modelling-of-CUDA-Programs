import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Configuration ---
# Ensure this matches the OUT_FILE from your Bash script
CSV_FILE = 'Results/H100/Latency/latency_summary.csv'
OUTPUT_IMAGE = 'Results/H100/Latency/latency_heatmap.png'
PLOT_TITLE = 'H100 Mean Memory Latency (Cycles) vs. Occupancy'
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
    # 1. Remove any 'FAIL' rows based on the MeanLatency column
    #    Note: We use 'MeanLatency_cycles' now, as per Volkov's method
    if 'MeanLatency_cycles' not in df.columns:
        print("Error: Column 'MeanLatency_cycles' not found in CSV.")
        return

    df = df[df['MeanLatency_cycles'].astype(str).str.lower() != 'fail']
    
    # 2. Convert columns to numeric types
    df['ThreadsPerBlock'] = pd.to_numeric(df['ThreadsPerBlock'])
    df['Shmem_KB'] = pd.to_numeric(df['Shmem_KB'])
    df['MeanLatency_cycles'] = pd.to_numeric(df['MeanLatency_cycles'])

    if df.empty:
        print("Error: No valid data found after cleaning.")
        return

    # Find the "Unloaded Latency" (The Minimum of the Means)
    min_mean_lat = df['MeanLatency_cycles'].min()
    
    # Find the configuration that achieved this minimum
    best_row = df.loc[df['MeanLatency_cycles'].idxmin()]
    print(f"------------------------------------------------")
    print(f"Unloaded Latency (Minimum of Means): {min_mean_lat:.2f} cycles")
    print(f"Achieved at: TPB={int(best_row['ThreadsPerBlock'])}, Shmem={int(best_row['Shmem_KB'])} KB")
    print(f"------------------------------------------------")

    # Pivot the data into a 2D grid for the heatmap
    try:
        pivot_table = df.pivot(
            index='ThreadsPerBlock', 
            columns='Shmem_KB', 
            values='MeanLatency_cycles' # Plotting the Mean
        )
    except ValueError as e:
        print(f"Error pivoting data: {e}")
        return

    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Draw the heatmap
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".1f", 
        cmap="viridis_r",  # '_r' reverses: Blue (low latency) is good, Yellow (high) is bad
        linewidths=.5,
        cbar_kws={'label': 'Mean Warp Latency (cycles)'}
    )
    
    # Highlight the cell with the minimum value
    # (Optional visual flair to spot the "Unloaded" value instantly)
    col_idx = pivot_table.columns.get_loc(best_row['Shmem_KB'])
    row_idx = pivot_table.index.get_loc(best_row['ThreadsPerBlock'])
    
    # Add a red rectangle around the best configuration
    rect = patches.Rectangle((col_idx, row_idx), 1, 1, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Set titles and labels
    ax.set_title(PLOT_TITLE, fontsize=16, pad=20)
    ax.set_xlabel("Shared Memory per Block (KB)", fontsize=12)
    ax.set_ylabel("Threads Per Block", fontsize=12)
    
    # Ensure directory exists for the image
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    
    print(f"\nSuccess! Plot saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    create_plot()