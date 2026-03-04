"""
Hong & Kim vs. Volkov Model Comparison — CORRECTED (Figure 7.1-style)
======================================================================

Fix summary
-----------
OLD code:  hong_kim_curve() called predict_hong_kim() which multiplied
           exec_cycles by num_rep.  Because active_blocks_per_sm ∝ n,
           num_rep ∝ 1/n, so the product was constant → flat curve.

NEW code:  hong_kim_ops_per_cycle_sm() models ONE SM directly:
             flops  = n × 32 × iterations × ai
             cycles = H&K Cases 1-3  (no num_rep)
           This gives the correct rising-then-plateauing shape.

Usage
-----
  python compare_models_plot_fixed.py --ai 32
  python compare_models_plot_fixed.py --all
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)
from hong_kim_model_fixed import (
    GPU_CONFIGS, GPUConfig,
    hong_kim_curve, volkov_curve,
)

ALL_AI = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]



# ── unit conversion ────────────────────────────────────────────────────────

def gflops_to_ops_per_cycle_sm(gflops_arr, gpu: GPUConfig):
    """GFLOPS → FMUL ops / cycle / SM  (= adds/cycle/SM in Volkov's notation)."""
    return np.asarray(gflops_arr) / (gpu.clock_ghz * gpu.n_sms)


# ══════════════════════════════════════════════════════════════════════════
#  Style
# ══════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        'font.family'      : 'serif',
        'font.serif'       : ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset' : 'stix',
        'font.size'        : 13,
        'axes.labelsize'   : 14,
        'legend.fontsize'  : 11,
        'xtick.labelsize'  : 11,
        'ytick.labelsize'  : 11,
        'axes.linewidth'   : 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size' : 4,
        'ytick.major.size' : 4,
        'xtick.direction'  : 'in',
        'ytick.direction'  : 'in',
    })


# ══════════════════════════════════════════════════════════════════════════
#  Plot
# ══════════════════════════════════════════════════════════════════════════

PANEL_LABELS = ["Hong & Kim 2009", "Volkov Model"]


def generate_comparison_plot(ai: int, df: pd.DataFrame, out_path: str):
    """Side-by-side figure matching Volkov's Figure 7.1 style."""
    setup_style()

    occ  = df["MaxAttainedOccupancy"].values
    gflo = df["GFLOPS"].values   # raw GFLOPS from CSV

    max_occ = occ.max()
    n_vals  = np.linspace(0.25, max_occ * 1.15, 600)

    # model curves in GFLOPS  (ops/cycle/SM × clock_GHz × n_SMs)
    def hk_gflops(n_arr):
        ops = hong_kim_curve(n_arr, GPU, ai, iterations=1000)
        return ops * GPU.clock_ghz * GPU.n_sms

    def vk_gflops(n_arr):
        return volkov_curve(n_arr, GPU, ai) * GPU.clock_ghz * GPU.n_sms

    hk_pred = hk_gflops(n_vals)
    vk_pred = vk_gflops(n_vals)

    y_max = max(gflo.max(), hk_pred.max(), vk_pred.max()) * 1.15
    xlim  = (0, max_occ * 1.15)
    ylim  = (0, y_max)

    fig = plt.figure(figsize=(9, 3.8), dpi=300)
    gs  = gridspec.GridSpec(1, 2, wspace=0.30)

    for col, (curve, label) in enumerate(zip([hk_pred, vk_pred], PANEL_LABELS)):
        ax = fig.add_subplot(gs[0, col])

        # Model curve
        ax.plot(n_vals, curve,
                color='#555555', linewidth=2.5, linestyle='-', zorder=2)
        # Experimental scatter
        ax.scatter(occ, gflo,
                   marker='o', s=28,
                   color='black', facecolors='black', zorder=3)

        ax.set_xlabel("warps/SM")
        if col == 0:
            ax.set_ylabel("GFLOP/s")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='-', linewidth=0.35,
                color='#bbbbbb', alpha=0.6)

        ax.text(0.97, 0.06, label,
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=11, fontstyle='italic')

    fig.tight_layout(pad=0.6)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_single(ai: int, gpu_name: str, gpu_config: GPUConfig):
    csv_dir = os.path.join(ROOT, "Results", gpu_name, "synthetic_streaming")
    out_dir = os.path.join(SCRIPT_DIR, gpu_name)
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(csv_dir, f"results_a{ai}.csv")
    if not os.path.exists(csv_path):
        print(f"❌  CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path).sort_values("MaxAttainedOccupancy")
    if df.empty:
        print(f"⚠️  Empty CSV for AI={ai}, skipping.")
        return
    out = os.path.join(out_dir, f"hk_vs_volkov_fixed_a{ai}_{gpu_name}.png")
    
    global GPU
    GPU = gpu_config
    generate_comparison_plot(ai, df, out)


def run_all(gpu_name: str, gpu_config: GPUConfig):
    found = 0
    csv_dir = os.path.join(ROOT, "Results", gpu_name, "synthetic_streaming")
    for ai in ALL_AI:
        csv_path = os.path.join(csv_dir, f"results_a{ai}.csv")
        if os.path.exists(csv_path):
            run_single(ai, gpu_name, gpu_config)
            found += 1
        else:
            print(f"⚠️  Skipping AI={ai}: CSV not found")
    print(f"\n✅  Generated plots for {found}/{len(ALL_AI)} AI values on {gpu_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H&K vs Volkov comparison plots — CORRECTED")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ai",  type=int)
    grp.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=str, required=True, choices=["RTX5000", "A100", "H100"], help="Target GPU")
    args = parser.parse_args()
    
    gpu_config = GPU_CONFIGS[args.gpu]
    
    if args.all:
        run_all(args.gpu, gpu_config)
    else:
        run_single(args.ai, args.gpu, gpu_config)
