"""
Zhang & Owens (HPCA 2011) Analytical GPU Performance Model
===========================================================

Reference: "A Quantitative Performance Analysis Model for GPU Architectures"
           Yao Zhang, John D. Owens, HPCA 2011.
           Described in: Volkov, Ph.D. Thesis 2016, §7.1, Figs 7.4, 7.5, 7.6.

MODEL SUMMARY
─────────────
Z&O microbenchmarks two occupancy-dependent throughput curves:

  alu(n)  — throughput of a pure-FP32 kernel  [adds/cycle/SM]
  mem(n)  — throughput of a pure-load kernel  [warp-loads/cycle/SM]

For a kernel with arithmetic intensity α (α ALU ops per 1 MEM op):

  Throughput = min( alu(n),  α · mem(n) )     [adds/cycle/SM]   ← Z&O (max overlap)
  Throughput = alu(n) · α·mem(n)              [harmonic mean]   ← "if adding" (no overlap)
               ───────────────────
               alu(n) + α·mem(n)

DERIVING alu(n) AND mem(n) FROM MICRO-KERNEL PARAMETERS
────────────────────────────────────────────────────────
alu(n)  — two regimes (Little's Law + HW peak):
  • Latency-bound  : n warps each stall alu_lat cycles  →  n·W / alu_lat  adds/cy/SM
  • Throughput-bound: fp32_cores_per_sm  adds/cy/SM (all ALU units busy)

mem(n)  — two regimes (Little's Law + bandwidth peak):
  • Latency-bound  : n warps each stall mem_lat_cycles  →  n·W / mem_lat  loads/cy/SM
  • BW-bound       : BW_per_SM / (W · bytes/access)  loads/cy/SM

Consistent units: both alu(n) and α·mem(n) are in ops/cycle/SM so
min(alu(n), α·mem(n)) directly gives add-throughput in adds/cycle/SM.
"""

import numpy as np
from typing import Dict, List, Union
from hong_kim_model_fixed import GPUConfig, GPU_CONFIGS, volkov_curve


# ─── core single-point function ───────────────────────────────────────────────

def zhang_owens_ops_per_cycle_sm(
    gpu: GPUConfig,
    ai: float,   # arithmetic intensity α  (ALU ops per MEM op)
    n: float,    # warps / SM  (occupancy)
) -> Dict:
    """
    Returns Z&O predicted throughput in adds/cycle/SM for n warps/SM.

    alu(n)  [adds/cycle/SM]
    ──────
    • Latency-bound : n·W / alu_lat
    • HW-peak-bound : fp32_cores_per_sm   ← number of FP32 ALU lanes on the SM

    mem(n)  [loads/cycle/SM, where 1 load = 1 warp × W threads × bytes_per_access]
    ──────
    • Latency-bound : n·W / mem_lat_cycles
    • BW-bound      : (BW_per_SM in bytes/cycle) / (W × bytes_per_access)

    Z&O formula  : Throughput = min( alu(n),  α·mem(n) )
    If-adding    : Throughput = harmonic-mean( alu(n), α·mem(n) )
    """
    n = max(float(n), 1e-9)
    W = gpu.warp_size

    # ── alu(n): adds/cycle/SM ─────────────────────────────────────────────
    # Latency-bound: n warps × W threads / alu pipeline latency
    alu_lat_bound  = n * W / gpu.alu_lat
    # Peak: the SM has fp32_cores_per_sm ALU lanes — all can fire per cycle
    alu_peak_bound = float(gpu.fp32_cores_per_sm)
    alu_n = min(alu_lat_bound, alu_peak_bound)

    # ── mem(n): warp-loads/cycle/SM → converted via factor W to add-scale ─
    # Latency-bound: n warps each waiting mem_lat_cycles
    mem_lat_bound  = n * W / gpu.mem_lat_cycles
    # BW-bound: per-SM share of total peak bandwidth
    bw_bytes_per_cycle  = (gpu.mem_bw_gbps * 1e9) / (gpu.clock_ghz * 1e9)
    bw_per_sm_per_cycle = bw_bytes_per_cycle / gpu.n_sms          # bytes/cy/SM
    mem_peak_bound = bw_per_sm_per_cycle / (W * gpu.load_bytes_per_access)
    mem_n = min(mem_lat_bound, mem_peak_bound)                    # loads/cy/SM

    # ── Z&O: Throughput = min( alu(n), α·mem(n) ) [adds/cy/SM] ───────────
    # α loads × W threads × ai adds/load = ai·W adds , but alu_n is already in
    # adds/cy/SM counted as individual thread-ops.  mem_n (loads/cy/SM) each
    # load services W threads; α·mem_n gives the same per-SM add rate.
    alpha_mem_n   = ai * mem_n
    zo_throughput = min(alu_n, alpha_mem_n)

    # ── "If adding": harmonic mean (no overlap) ───────────────────────────
    denom = alu_n + alpha_mem_n
    if_adding = (alu_n * alpha_mem_n) / denom if denom > 0 else 0.0

    return {
        "ops_per_cycle_sm":           zo_throughput,
        "if_adding_ops_per_cycle_sm": if_adding,
        "alu_n":                      alu_n,
        "mem_n":                      mem_n,
        "alpha_mem_n":                alpha_mem_n,
        "alu_lat_bound":              alu_lat_bound,
        "alu_peak_bound":             alu_peak_bound,
        "mem_lat_bound":              mem_lat_bound,
        "mem_peak_bound":             mem_peak_bound,
    }


# ─── vectorised helpers ───────────────────────────────────────────────────────

def zhang_owens_curve(
    n_arr: Union[List[float], np.ndarray],
    gpu:  GPUConfig,
    ai:   float,
) -> np.ndarray:
    """Z&O predicted throughput (adds/cycle/SM) as a function of warps/SM."""
    return np.array([
        zhang_owens_ops_per_cycle_sm(gpu, ai, n)["ops_per_cycle_sm"]
        for n in n_arr
    ])


def if_adding_curve(
    n_arr: Union[List[float], np.ndarray],
    gpu:  GPUConfig,
    ai:   float,
) -> np.ndarray:
    """'If adding' (no overlap) comparison curve, same axes."""
    return np.array([
        zhang_owens_ops_per_cycle_sm(gpu, ai, n)["if_adding_ops_per_cycle_sm"]
        for n in n_arr
    ])


# ─── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    gpu_name = sys.argv[1] if len(sys.argv) > 1 else "RTX5000"
    gpu = GPU_CONFIGS[gpu_name]

    print(f"\nGPU : {gpu.name}")
    print(f"      fp32_cores/SM={gpu.fp32_cores_per_sm}  "
          f"alu_lat={gpu.alu_lat}  mem_lat={gpu.mem_lat_cycles} cy  "
          f"mem_bw={gpu.mem_bw_gbps} GB/s  n_sms={gpu.n_sms}")

    for ai in [2, 4, 16, 32, 64]:
        print(f"\n{'─'*74}")
        print(f"  AI = {ai}  (α = {ai} FP32 adds per memory load)")
        hdr_n    = f"{'n':>5}"
        hdr_zo   = f"{'Z&O [adds/cy/SM]':>18}"
        hdr_ia   = f"{'if-adding':>14}"
        hdr_vk   = f"{'Volkov':>14}"
        hdr_alu  = f"{'alu(n)':>12}"
        hdr_amem = f"{'α·mem(n)':>12}"
        print(f"  {hdr_n} {hdr_alu} {hdr_amem} {hdr_zo} {hdr_ia} {hdr_vk}")
        print(f"  {'─'*74}")

        n_arr = np.array([1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32])
        zo  = zhang_owens_curve(n_arr, gpu, ai)
        ia  = if_adding_curve  (n_arr, gpu, ai)
        vk  = volkov_curve     (n_arr, gpu, ai)
        raw = [zhang_owens_ops_per_cycle_sm(gpu, ai, n) for n in n_arr]

        for i, nn in enumerate(n_arr):
            r = raw[i]
            print(f"  {int(nn):>5} {r['alu_n']:>12.2f} {r['alpha_mem_n']:>12.2f}"
                  f" {zo[i]:>18.2f} {ia[i]:>14.2f} {vk[i]:>14.2f}")
