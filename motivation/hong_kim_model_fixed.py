"""
Hong & Kim (ISCA 2009) Analytical GPU Performance Model — CORRECTED
=====================================================================

FIX: The original implementation multiplied exec_cycles by num_rep and
computed total_flops via num_blocks. Due to how active_blocks_per_sm
scales with n, these factors **algebraically cancel**, making the predicted
curve flat across all occupancies — completely masking the rising-then-
plateauing shape visible in Volkov's Figure 7.1.

Root cause:
    exec_cycles ∝ (N / MWP) × num_rep          [grows with n via Case-2 N term]
    num_rep = num_blocks / (active_blk/SM × SMs)  [shrinks with n proportionally]
    → product is constant in n  →  flat GFLOPS prediction

Fix: compute per-SM throughput directly from n warps (one SM, one pass),
without the num_blocks / num_rep scheduling layer.  The per-SM formula is:
    flops_per_SM = n × warp_size × iterations × ai
    exec_cycles  = H&K formula (Cases 1-3) WITHOUT num_rep
    ops/cycle/SM = flops_per_SM / exec_cycles

References:
  Hong & Kim, ISCA 2009 — Eqs (8)-(24)
  Volkov, Ph.D. thesis 2016, §7.1, Fig 7.1
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


# ── GPU configurations ─────────────────────────────────────────────────────

@dataclass
class GPUConfig:
    name: str
    n_sms: int
    clock_ghz: float
    mem_bw_gbps: float
    mem_lat_cycles: int
    issue_cycles_per_inst: int
    issue_thru: float                 # warp schedulers per SM (used by H&K)
    fp32_cores_per_sm: int = 64       # hardware FP32 ALU lanes per SM
                                      # = peak adds/cycle/SM  (used by Z&O alu(n))
    alu_lat: int = 4
    warp_size: int = 32
    departure_delay: int = 4
    load_bytes_per_access: int = 4   # sizeof(uint32_t) — 4-byte loads


GPU_CONFIGS = {
    "RTX5000": GPUConfig(
        name="Quadro RTX 5000",
        n_sms=48,
        clock_ghz=1.815,
        mem_bw_gbps=340.0,        # Verified
        mem_lat_cycles=236,       # Verified
        issue_cycles_per_inst=1,  # 64 FP32 / 2 schedulers → 1 cycle/issue
        issue_thru=2.0,           # 2 warp schedulers per SM
        fp32_cores_per_sm=64,     # 3072 CUDA cores / 48 SMs
    ),
    "A100": GPUConfig(
        name="A100",
        n_sms=108,
        clock_ghz=1.410,
        mem_bw_gbps=1265.0,
        mem_lat_cycles=240,
        issue_cycles_per_inst=2,
        issue_thru=2.0,
        fp32_cores_per_sm=64,     # 6912 CUDA cores / 108 SMs
    ),
    "H100": GPUConfig(
        name="H100",
        n_sms=114,
        clock_ghz=1.755,
        mem_bw_gbps=1585.0,
        mem_lat_cycles=352,
        issue_cycles_per_inst=1,
        issue_thru=4.0,
        fp32_cores_per_sm=128,    # 14592 CUDA cores / 114 SMs
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  Hong & Kim — PER-SM THROUGHPUT  (corrected)
# ══════════════════════════════════════════════════════════════════════════

def hong_kim_ops_per_cycle_sm(
    gpu: GPUConfig,
    ai: int,
    n: float,           # warps / SM  (can be fractional for smooth curve)
    iterations: int = 1000,
) -> Dict:
    """
    Return H&K predicted throughput in FMUL ops/cycle/SM for n warps/SM.

    Models ONE SM running n warps for `iterations` loop iterations,
    each iteration = 1 memory load + ai FMULs.

    Key difference from the buggy predict_hong_kim():
      • No num_blocks / num_rep — those factors cancel algebraically,
        producing a spuriously flat curve.
      • flops = n × warp_size × iterations × ai   (one SM's worth)
      • exec_cycles = H&K formula (no num_rep multiplier)
    """
    N = max(1.0, float(n))   # continuous n — H&K equations have no integer-warp assumption
    clock_hz = gpu.clock_ghz * 1e9

    # ── Step 1: instruction counts (one warp, all iterations) ──
    num_mem_insts   = iterations
    num_total_insts = iterations * (ai + 1)   # 1 load + ai FMULs per iter

    # ── Step 2: per-warp cycle costs ──
    # H&K uses issue latency only for Comp_cycles (no alu_lat pipeline depth)
    Comp_cycles = num_total_insts * gpu.issue_cycles_per_inst
    Mem_L       = gpu.mem_lat_cycles
    Mem_cycles  = Mem_L * num_mem_insts

    # Computation period between consecutive memory instructions (Eq 2)
    Comp_p = Comp_cycles / num_mem_insts

    # ── Step 3: MWP (per SM) ──
    MWP_Without_BW = Mem_L / gpu.departure_delay   # Eq (16)/(17)

    bytes_per_mem_inst  = gpu.warp_size * gpu.load_bytes_per_access  # 128 B
    bw_per_cycle_per_sm = (gpu.mem_bw_gbps * 1e9) / (gpu.clock_ghz * 1e9 * gpu.n_sms)
    BW_per_warp         = bytes_per_mem_inst / Mem_L   # bytes/cycle consumed by one in-flight warp
    MWP_peak_BW         = bw_per_cycle_per_sm / BW_per_warp   # Eq (6)/(7) per-SM form

    MWP = min(MWP_Without_BW, MWP_peak_BW, N)
    MWP = max(MWP, 1.0)

    # ── Step 4: CWP ──
    CWP_full = (Mem_cycles + Comp_cycles) / Comp_cycles   # Eq (8)/(9)
    CWP = min(CWP_full, N)

    # ── Step 5: exec_cycles for N warps on ONE SM (NO num_rep) ──
    if CWP >= MWP:
        if MWP >= N:
            # Case 1: sufficient parallelism — memory & compute fully overlap
            exec_cycles = Mem_cycles + Comp_cycles + Comp_p * (MWP - 1)
            case = "sufficient"
        else:
            # Case 2: memory-bound
            exec_cycles = Mem_cycles * (N / MWP) + Comp_p * (MWP - 1)
            case = "memory_bound"
    else:
        # Case 3: compute-bound
        exec_cycles = Mem_L + Comp_cycles * N
        case = "compute_bound"

    if exec_cycles <= 0:
        return {"ops_per_cycle_sm": 0.0, "case": case}

    # ── Step 6: per-SM FMUL ops / cycle ──
    flops_per_sm     = N * gpu.warp_size * iterations * ai
    ops_per_cycle_sm = flops_per_sm / exec_cycles

    return {
        "ops_per_cycle_sm": ops_per_cycle_sm,
        "case": case,
        "exec_cycles": exec_cycles,
        "MWP": MWP,
        "CWP": CWP,
        "MWP_Without_BW": MWP_Without_BW,
        "MWP_peak_BW": MWP_peak_BW,
        "Comp_cycles": Comp_cycles,
        "Mem_cycles": Mem_cycles,
    }


def hong_kim_curve(n_arr, gpu: GPUConfig, ai: int, iterations: int = 1000):
    """Vectorised H&K curve: ops/cycle/SM vs warps/SM array."""
    return np.array([
        hong_kim_ops_per_cycle_sm(gpu, ai, float(n), iterations)["ops_per_cycle_sm"]
        for n in n_arr
    ])


# ══════════════════════════════════════════════════════════════════════════
#  Volkov Model (for comparison)
# ══════════════════════════════════════════════════════════════════════════

def volkov_curve(n_arr, gpu: GPUConfig, ai: int):
    """Volkov model: ops/cycle/SM vs warps/SM (closed-form Little's Law)."""
    n = np.asarray(n_arr, dtype=float)
    mem_thru      = (gpu.mem_bw_gbps * 1e9) / (
                        gpu.clock_ghz * 1e9 * gpu.n_sms
                        * gpu.warp_size * gpu.load_bytes_per_access)
    latency_bound = n / (gpu.mem_lat_cycles + ai * gpu.alu_lat)
    memory_bound  = np.full_like(n, mem_thru)
    issue_bound   = np.full_like(n, gpu.issue_thru / (ai + 1))
    T_m = np.minimum.reduce([latency_bound, memory_bound, issue_bound])
    return gpu.warp_size * ai * T_m   # ops/cycle/SM


# ══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gpu = GPU_CONFIGS["RTX5000"]
    for ai in [2, 8, 32, 64]:
        print(f"\n=== {gpu.name}  AI={ai} ===")
        print(f"  MWP_peak_BW = {hong_kim_ops_per_cycle_sm(gpu, ai, 1)['MWP_peak_BW']:.2f}  "
              f"MWP_Without_BW = {hong_kim_ops_per_cycle_sm(gpu, ai, 1)['MWP_Without_BW']:.2f}")
        print(f"{'n':>6} {'H&K ops/cy/SM':>15} {'Volkov ops/cy/SM':>18} {'case':>14}")
        print("-" * 58)
        for n in [1, 2, 4, 6, 8, 10, 12, 16, 24, 32]:
            hk = hong_kim_ops_per_cycle_sm(gpu, ai, n)
            vk = volkov_curve([n], gpu, ai)[0]
            print(f"{n:>6}  {hk['ops_per_cycle_sm']:>14.2f}  {vk:>17.2f}  {hk['case']:>14}")
