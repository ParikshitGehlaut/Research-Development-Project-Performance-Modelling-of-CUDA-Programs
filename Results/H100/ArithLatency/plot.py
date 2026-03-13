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
PLOT_TITLE = f'{TARGET_OP} Mean Latency vs. Occupancy'

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
    
    # Mathematical perfect scale: 10 cols * 1.0" + 3.0" margin | 4 rows * 1.2" + 3.5" margin
    FIG_W, FIG_H = 13.0, 8.3
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=350)

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",               # 1 decimal: "4.1" not "4.06" — fits in narrower cells
        cmap="viridis_r",
        linewidths=0.8,
        linecolor="gray",
        square=False,
        annot_kws={"size": 28, "weight": "bold"},
        ax=ax,
        cbar=False,
    )

    # Highlight minimum latency cell
    col_idx = pivot_table.columns.get_loc(best_row['Shmem_KB'])
    row_idx = pivot_table.index.get_loc(best_row['ThreadsPerBlock'])
    rect = patches.Rectangle(
        (col_idx, row_idx), 1, 1,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Labels and title
    ax.set_title(PLOT_TITLE, fontsize=28, pad=15)
    ax.set_xlabel("Shared Memory per Block (KB)", fontsize=32, fontweight='bold', labelpad=15)
    ax.set_ylabel("Threads Per Block", fontsize=32, fontweight='bold', labelpad=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=26, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=26, rotation=0)

    # Exact mathematically aligned margins for (13x8.3 overall -> 10x4.8 plot area + 2.5 L / 0.5 R / 1.5 T / 2.0 B)
    plt.subplots_adjust(left=0.1923, right=0.9615, top=0.8193, bottom=0.2410)
    
    # Ensure output directory exists (though it's same as SCRIPT_DIR now)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Removed bbox_inches='tight' since it arbitrarily breaks absolute margin alignments.
    plot_path = os.path.join(OUTPUT_DIR, f'alu_latency_heatmap_h100.png')
    plt.savefig(plot_path, dpi=350)
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
