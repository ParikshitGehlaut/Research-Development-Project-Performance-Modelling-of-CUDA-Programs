import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ------------------------------
# Configuration
# ------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, 'arith_latency_summary.csv')
OUTPUT_DIR = SCRIPT_DIR  # Output to the same directory as the script
TARGET_OP = 'FMUL'  # Change to 'FMUL', 'FMA', etc. as needed
PLOT_TITLE = f'H100 {TARGET_OP} Mean Latency (Cycles) vs. Occupancy'

# ------------------------------
def create_plot():
    print(f"Reading data from {CSV_FILE}...")
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: File not found at {CSV_FILE}")
        print("Please ensure the CSV file exists.")
        return
    
    # Read CSV
    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        print(f"Error: {CSV_FILE} is empty.")
        return
    
    # Validate required columns
    required_cols = ['OpType', 'ThreadsPerBlock', 'Shmem_KB', 'MeanLatency_cycles']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns. Expected: {required_cols}")
        return
    
    # Filter for target operation
    df_op = df[df['OpType'] == TARGET_OP].copy()
    
    if df_op.empty:
        print(f"Error: No data found for operation '{TARGET_OP}'")
        return
    
    # Convert types
    df_op['ThreadsPerBlock'] = pd.to_numeric(df_op['ThreadsPerBlock'])
    df_op['Shmem_KB'] = pd.to_numeric(df_op['Shmem_KB'])
    df_op['MeanLatency_cycles'] = pd.to_numeric(df_op['MeanLatency_cycles'])
    
    # Identify minimum latency configuration
    min_mean_lat = df_op['MeanLatency_cycles'].min()
    best_row = df_op.loc[df_op['MeanLatency_cycles'].idxmin()]
    
    print("------------------------------------------------")
    print(f"Minimum {TARGET_OP} Latency: {min_mean_lat:.2f} cycles")
    print(f"Achieved at TPB={int(best_row['ThreadsPerBlock'])}, "
          f"Shmem={int(best_row['Shmem_KB'])} KB")
    print("------------------------------------------------")
    
    # Pivot to matrix format
    try:
        pivot_table = df_op.pivot(
            index='ThreadsPerBlock',
            columns='Shmem_KB',
            values='MeanLatency_cycles'
        )
    except ValueError as e:
        print(f"Pivot error: {e}")
        return
    
    # ------------------------------
    # Publication-Ready Heatmap
    # ------------------------------
    plt.figure(figsize=(6.0, 3.8), dpi=300)  # Compact for 2-column papers
    
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",  # 2 decimal places for precision
        cmap="viridis_r",  # Reversed: blue = low latency (good)
        linewidths=0.3,
        linecolor="gray",
        square=True,
        annot_kws={"size": 8, "weight": "normal"},
        cbar_kws={
            'label': 'Mean Latency (cycles)',
            'shrink': 0.75
        }
    )
    
    # Highlight minimum latency cell
    col_idx = pivot_table.columns.get_loc(best_row['Shmem_KB'])
    row_idx = pivot_table.index.get_loc(best_row['ThreadsPerBlock'])
    rect = patches.Rectangle(
        (col_idx, row_idx), 1, 1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # Labels and title (optimized for paper figures)
    ax.set_title(PLOT_TITLE, fontsize=11, pad=6)
    ax.set_xlabel("Shared Memory per Block (KB)", fontsize=9)
    ax.set_ylabel("Threads Per Block", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    
    # Adjust colorbar label font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Mean Latency (cycles)', fontsize=9)
    
    plt.tight_layout()
    
    # Ensure output directory exists (though it's same as SCRIPT_DIR now)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save formatted plot
    plot_path = os.path.join(OUTPUT_DIR, f'{TARGET_OP}_latency_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Save the filtered data to CSV in the same directory
    csv_out_path = os.path.join(OUTPUT_DIR, f'{TARGET_OP}_latency_summary.csv')
    df_op.to_csv(csv_out_path, index=False)
    print(f"CSV data saved to {csv_out_path}")
    
    print("\nSuccess!")
    
    # Optionally show the plot
    # plt.show()

if __name__ == "__main__":
    create_plot()
