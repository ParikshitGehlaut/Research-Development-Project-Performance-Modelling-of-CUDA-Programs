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
PLOT_TITLE = 'Mean Memory Latency vs. Occupancy'
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

    # Mathematical perfect scale: 7 cols * 1.0" + 3.0" margin | 5 rows * 1.2" + 3.5" margin
    FIG_W, FIG_H = 10.0, 9.5
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=350)

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".0f",               # integer display
        cmap="viridis_r",
        linewidths=0.8,
        linecolor="gray",
        square=False,  # square implicitly enforced by precise subplots_adjust
        annot_kws={"size": 28, "weight": "bold"},
        ax=ax,
        cbar=False,              # removed — frees ~15% width for larger fonts
    )

    # Highlight minimum latency
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

    # Exact mathematically aligned margins for (10x9.5 overall -> 7x6 plot area + 2.5 L / 0.5 R / 1.5 T / 2.0 B)
    plt.subplots_adjust(left=0.25, right=0.95, top=0.8421, bottom=0.2105)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_IMAGE) if os.path.dirname(OUTPUT_IMAGE) else '.', exist_ok=True)

    # Removed bbox_inches='tight' since it arbitrarily breaks absolute margin alignments.
    plt.savefig(OUTPUT_IMAGE, dpi=350)

    print(f"\nSuccess! Plot saved to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    create_plot()
