"""
Compare Prior Analytical Models vs Measured Data
=================================================

Reads the existing synthetic kernel (streaming memory) CSV results and
overlays predictions from:
  1. Hong & Kim (ISCA 2009) — MWP/CWP model
  2. Volkov (2016) — Little's Law model
  3. Measured (from kernel runs)

Usage:
  python compare_models.py --gpu RTX5000
  python compare_models.py --gpu H100
  python compare_models.py --gpu A100

Reads data from: Results/<GPU>/simplistic_with_streaming_memory/
Writes plots to: Results/<GPU>/model_comparison/
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# Add parent dir to path so we can import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hong_kim_model import GPU_CONFIGS, predict_hong_kim, predict_volkov

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not found. Will print results to stdout only.")


def load_measured_data(data_dir: str, ai_values: list) -> dict:
    """
    Load measured GFLOPS from CSV files.

    Returns: {ai: [(threads_per_block, shmem_kb, gflops, occupancy), ...]}
    """
    data = {}
    for ai in ai_values:
        filename = os.path.join(data_dir, f"results_a{ai}.csv")
        if not os.path.exists(filename):
            print(f"  Skipping AI={ai}, file not found: {filename}")
            continue

        rows = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tpb = int(row['THREADS_PER_BLOCK'])
                    shmem = int(row['SHMEM_KB'])
                    gflops = float(row['GFLOPS'])
                    occ = int(row['MaxAttainedOccupancy'])
                    rows.append((tpb, shmem, gflops, occ))
                except (ValueError, KeyError):
                    continue
        data[ai] = rows
    return data


def find_peak_per_occupancy(rows):
    """
    For each occupancy level, find the peak measured GFLOPS.
    Returns: {occupancy: peak_gflops}
    """
    occ_to_gflops = defaultdict(float)
    for tpb, shmem, gflops, occ in rows:
        if gflops > occ_to_gflops[occ]:
            occ_to_gflops[occ] = gflops
    return dict(sorted(occ_to_gflops.items()))


def run_comparison(gpu_name: str, data_dir: str, output_dir: str,
                   iterations: int = 500, num_blocks: int = 1000):
    """Run both models and compare against measured data."""

    gpu = GPU_CONFIGS[gpu_name]
    ai_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\n{'='*70}")
    print(f"  Model Comparison: {gpu.name}")
    print(f"{'='*70}")

    measured_data = load_measured_data(data_dir, ai_values)
    if not measured_data:
        print("ERROR: No measured data found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Per-AI comparison at peak occupancy ──
    summary_rows = []

    for ai in sorted(measured_data.keys()):
        rows = measured_data[ai]
        occ_peaks = find_peak_per_occupancy(rows)

        # Find the overall peak measured GFLOPS and its occupancy
        peak_occ = max(occ_peaks, key=occ_peaks.get)
        peak_measured = occ_peaks[peak_occ]

        # Use the peak-performing occupancy configuration for model predictions
        # Find the TPB that achieved peak
        peak_tpb = 256  # default
        for tpb, shmem, gflops, occ in rows:
            if occ == peak_occ and abs(gflops - peak_measured) < 0.01:
                peak_tpb = tpb
                break

        hk = predict_hong_kim(gpu, ai, peak_occ, iterations,
                              num_blocks, peak_tpb)
        vk = predict_volkov(gpu, ai, peak_occ, iterations,
                            num_blocks, peak_tpb)

        hk_error = ((hk['gflops'] - peak_measured) / peak_measured * 100
                     if peak_measured > 0 else 0)
        vk_error = ((vk['gflops'] - peak_measured) / peak_measured * 100
                     if peak_measured > 0 else 0)

        summary_rows.append({
            'ai': ai, 'occ': peak_occ, 'tpb': peak_tpb,
            'measured': peak_measured,
            'hong_kim': hk['gflops'], 'hk_error': hk_error, 'hk_case': hk['case'],
            'volkov': vk['gflops'], 'vk_error': vk_error, 'vk_bound': vk['bound'],
        })

    # ── Print summary table ──
    print(f"\n{'AI':>4} {'Occ':>4} {'Measured':>10} "
          f"{'H&K':>10} {'H&K Err%':>9} {'H&K Case':>14} "
          f"{'Volkov':>10} {'Vk Err%':>9} {'Vk Bound':>10}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r['ai']:>4} {r['occ']:>4} {r['measured']:>10.1f} "
              f"{r['hong_kim']:>10.1f} {r['hk_error']:>+8.1f}% {r['hk_case']:>14} "
              f"{r['volkov']:>10.1f} {r['vk_error']:>+8.1f}% {r['vk_bound']:>10}")

    # ── Save summary CSV ──
    csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {csv_path}")

    # ── Generate plots ──
    if HAS_MATPLOTLIB:
        _plot_comparison(summary_rows, gpu, output_dir)
        _plot_per_ai_occupancy_sweep(measured_data, gpu, iterations,
                                      num_blocks, output_dir)

    return summary_rows


def _plot_comparison(summary_rows, gpu, output_dir):
    """Plot peak GFLOPS vs AI: Measured, Hong & Kim, Volkov."""
    ais = [r['ai'] for r in summary_rows]
    measured = [r['measured'] for r in summary_rows]
    hong_kim = [r['hong_kim'] for r in summary_rows]
    volkov = [r['volkov'] for r in summary_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: GFLOPS comparison ──
    ax1.plot(ais, measured, 'ko-', label='Measured', linewidth=2, markersize=7)
    ax1.plot(ais, hong_kim, 'rs--', label='Hong & Kim (2009)',
             linewidth=1.5, markersize=6)
    ax1.plot(ais, volkov, 'b^--', label='Volkov (2016)',
             linewidth=1.5, markersize=6)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Arithmetic Intensity (AI)')
    ax1.set_ylabel('Peak GFLOPS')
    ax1.set_title(f'{gpu.name}: Peak GFLOPS vs AI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # ── Right: Error % ──
    hk_err = [r['hk_error'] for r in summary_rows]
    vk_err = [r['vk_error'] for r in summary_rows]
    ax2.bar([x - 0.15 for x in range(len(ais))], hk_err, width=0.3,
            color='red', alpha=0.7, label='Hong & Kim Error')
    ax2.bar([x + 0.15 for x in range(len(ais))], vk_err, width=0.3,
            color='blue', alpha=0.7, label='Volkov Error')
    ax2.set_xticks(range(len(ais)))
    ax2.set_xticklabels([str(a) for a in ais])
    ax2.set_xlabel('Arithmetic Intensity (AI)')
    ax2.set_ylabel('Prediction Error (%)')
    ax2.set_title(f'{gpu.name}: Model Prediction Error')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"model_comparison_{gpu.name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {path}")


def _plot_per_ai_occupancy_sweep(measured_data, gpu, iterations,
                                  num_blocks, output_dir):
    """For selected AI values, plot GFLOPS vs occupancy with model overlays."""
    selected_ais = [ai for ai in [4, 16, 64, 256] if ai in measured_data]
    if not selected_ais:
        return

    fig, axes = plt.subplots(1, len(selected_ais),
                             figsize=(5 * len(selected_ais), 4.5),
                             sharey=False)
    if len(selected_ais) == 1:
        axes = [axes]

    for ax, ai in zip(axes, selected_ais):
        rows = measured_data[ai]
        occ_peaks = find_peak_per_occupancy(rows)

        occs = sorted(occ_peaks.keys())
        meas_gflops = [occ_peaks[o] for o in occs]

        # Model predictions at each occupancy
        hk_gflops = []
        vk_gflops = []
        for occ in occs:
            # Find TPB for this occupancy
            tpb = 256
            for t, s, g, o in rows:
                if o == occ:
                    tpb = t
                    break
            hk = predict_hong_kim(gpu, ai, occ, iterations, num_blocks, tpb)
            vk = predict_volkov(gpu, ai, occ, iterations, num_blocks, tpb)
            hk_gflops.append(hk['gflops'])
            vk_gflops.append(vk['gflops'])

        ax.plot(occs, meas_gflops, 'ko-', label='Measured', markersize=5)
        ax.plot(occs, hk_gflops, 'rs--', label='Hong & Kim', markersize=4)
        ax.plot(occs, vk_gflops, 'b^--', label='Volkov', markersize=4)
        ax.set_xlabel('Occupancy (warps/SM)')
        ax.set_ylabel('Peak GFLOPS')
        ax.set_title(f'AI = {ai}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{gpu.name}: GFLOPS vs Occupancy', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f"occupancy_sweep_{gpu.name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Occupancy sweep plot saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Hong & Kim and Volkov models vs measured data")
    parser.add_argument('--gpu', required=True,
                        choices=['RTX5000', 'A100', 'H100'],
                        help='GPU to compare')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Kernel iterations (default: 500)')
    parser.add_argument('--blocks', type=int, default=1000,
                        help='Num blocks (default: 1000)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'Results', args.gpu,
                            'simplistic_with_streaming_memory')
    output_dir = os.path.join(base, 'Results', args.gpu, 'model_comparison')

    run_comparison(args.gpu, data_dir, output_dir,
                   args.iterations, args.blocks)


if __name__ == "__main__":
    main()
