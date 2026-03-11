import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ------------------------------
# Configuration
# ------------------------------
CSV_FILE = 'latency_summary.csv'
OUTPUT_IMAGE = 'mem_latency_heatmap_rtx5000.png'
PLOT_TITLE = 'RTX 5000 Mean Memory Latency (Cycles) vs. Occupancy'
# ------------------------------


def create_plot():
    print(f"Reading data from {CSV_FILE}...")

    if not os.path.exists(CSV_FILE):
        print(f"Error: File not found at {CSV_FILE}")
        print("Please run ./run_memory_latency.sh first.")
        return

    # Read CSV
    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        print(f"Error: {CSV_FILE} is empty.")
        return

    # Validate column
    if 'MeanLatency_cycles' not in df.columns:
        print("Error: Column 'MeanLatency_cycles' missing from CSV.")
        return

    # Clean invalid rows
    df = df[df['MeanLatency_cycles'].astype(str).str.lower() != 'fail']

    # Convert types
    df['ThreadsPerBlock'] = pd.to_numeric(df['ThreadsPerBlock'])
    df['Shmem_KB'] = pd.to_numeric(df['Shmem_KB'])
    df['MeanLatency_cycles'] = pd.to_numeric(df['MeanLatency_cycles'])

    if df.empty:
        print("Error: No valid data after cleaning.")
        return

    # Identify unloaded (minimum) latency configuration
    min_mean_lat = df['MeanLatency_cycles'].min()
    best_row = df.loc[df['MeanLatency_cycles'].idxmin()]

    print("------------------------------------------------")
    print(f"Unloaded Latency (Minimum Mean): {min_mean_lat:.2f} cycles")
    print(f"Achieved at TPB={int(best_row['ThreadsPerBlock'])}, "
          f"Shmem={int(best_row['Shmem_KB'])} KB")
    print("------------------------------------------------")

    # Pivot to matrix format
    try:
        pivot_table = df.pivot(
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

    plt.figure(figsize=(20, 14), dpi=300)  # Large figure so fonts survive 3-across shrink

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="viridis_r",
        linewidths=0.3,
        linecolor="gray",
        square=True,  # improves visual clarity when printed small
        annot_kws={"size": 32},
        cbar_kws={
            'label': 'Mean Warp Latency (cycles)',
            'shrink': 0.75
        }
    )

    # Highlight minimum latency
    col_idx = pivot_table.columns.get_loc(best_row['Shmem_KB'])
    row_idx = pivot_table.index.get_loc(best_row['ThreadsPerBlock'])

    rect = patches.Rectangle(
        (col_idx, row_idx), 1, 1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Labels and title (large fonts for 3-across layout in 2-column paper)
    ax.set_title(PLOT_TITLE, fontsize=44, pad=20)
    ax.set_xlabel("Shared Memory per Block (KB)", fontsize=36)
    ax.set_ylabel("Threads Per Block", fontsize=36)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=32)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=32)

    # Colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label('Mean Warp Latency (cycles)', fontsize=32)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_IMAGE) if os.path.dirname(OUTPUT_IMAGE) else '.', exist_ok=True)

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')

    print(f"\nSuccess! Plot saved to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    create_plot()
