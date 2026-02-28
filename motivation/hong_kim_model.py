"""
Hong & Kim (ISCA 2009) Analytical GPU Performance Model
========================================================

Implementation of the MWP/CWP model from:
  S. Hong and H. Kim, "An Analytical Model for a GPU Architecture with
  Memory-level and Thread-level Parallelism Awareness," ISCA 2009.

This implementation targets the streaming (coalesced) memory access pattern.

Key insight this model misses (and what we demonstrate):
  Hong & Kim compute Comp_cycles = #instructions × issue_latency.
  They do NOT account for arithmetic pipeline latency (alu_lat) in
  dependent instruction chains. On modern GPUs where alu_lat > issue_latency
  (e.g., 4 cycles vs 2 cycles on RTX 5000), this causes systematic
  overestimation of throughput at high arithmetic intensity.

References for departure delay:
  - Hong & Kim (ISCA 2009), Section 4: measured via microbenchmark
  - Mei & Chu (GPGPU 2017): ~4 cycles for coalesced global memory on Kepler/Maxwell
  - Jia et al. (HPCA 2018): 4-6 cycles for coalesced on Volta
"""

from dataclasses import dataclass
from typing import Dict, Optional
import math


@dataclass
class GPUConfig:
    """GPU architecture parameters."""
    name: str
    n_sms: int
    clock_ghz: float
    mem_bw_gbps: float          # Effective memory BW (measured)
    mem_lat_cycles: int         # Memory latency (measured via pointer chasing)
    issue_cycles_per_inst: int  # Cycles to issue one warp-instruction
    issue_thru: float           # Warp-instructions/cycle/SM (all schedulers)
    alu_lat: int = 4            # Arithmetic pipeline latency (measured)
    warp_size: int = 32
    departure_delay: int = 4    # Coalesced memory departure delay (see refs above)
    load_bytes_per_access: int = 8  # sizeof(uintptr_t)


# ── GPU configurations using OUR measured parameters ──────────────────────
GPU_CONFIGS = {
    "RTX5000": GPUConfig(
        name="Quadro RTX 5000",
        n_sms=48,
        clock_ghz=1.815,
        mem_bw_gbps=352.0,
        mem_lat_cycles=431,
        issue_cycles_per_inst=2,   # 32 / 16 FP32 cores per partition
        issue_thru=2.0,
    ),
    "A100": GPUConfig(
        name="A100",
        n_sms=108,
        clock_ghz=1.410,
        mem_bw_gbps=1265.0,
        mem_lat_cycles=240,
        issue_cycles_per_inst=2,
        issue_thru=2.0,
    ),
    "H100": GPUConfig(
        name="H100",
        n_sms=114,
        clock_ghz=1.755,
        mem_bw_gbps=1585.0,
        mem_lat_cycles=352,
        issue_cycles_per_inst=1,   # 32 / 32 FP32 cores per partition
        issue_thru=4.0,
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  Hong & Kim Model
# ══════════════════════════════════════════════════════════════════════════

def predict_hong_kim(
    gpu: GPUConfig,
    ai: int,
    n_warps_per_sm: int,
    iterations: int,
    num_blocks: int,
    threads_per_block: int,
) -> Dict:
    """
    Predict GFLOPS using Hong & Kim's MWP/CWP model (ISCA 2009).

    Parameters
    ----------
    gpu              : target GPU architecture
    ai               : arithmetic intensity (FMUL ops per memory load)
    n_warps_per_sm   : active warps per SM (measured MaxAttainedOccupancy)
    iterations       : loop iterations in the kernel
    num_blocks       : total thread blocks launched
    threads_per_block: threads per block

    Returns
    -------
    dict with gflops, case label, intermediate values for debugging.
    """
    if ai == 0 or n_warps_per_sm == 0:
        return {"gflops": 0.0, "case": "N/A"}

    N = n_warps_per_sm
    warps_per_block = threads_per_block // gpu.warp_size
    if warps_per_block == 0:
        return {"gflops": 0.0, "case": "N/A"}

    clock_hz = gpu.clock_ghz * 1e9

    # ── Step 1: Per-warp instruction counts ──
    num_mem_insts = iterations
    num_total_insts = iterations * (ai + 1)   # 1 load + AI FMULs per iter

    # ── Step 2: Per-warp cycle estimates ──
    # Eq (19): Comp_cycles — Hong & Kim use issue latency only (no alu_lat!)
    Comp_cycles = num_total_insts * gpu.issue_cycles_per_inst

    # Eq (11)+(18): Memory cycles (coalesced → Mem_L = Mem_LD)
    Mem_L = gpu.mem_lat_cycles
    Mem_cycles = Mem_L * num_mem_insts

    # Eq (2): Computation period between memory instructions
    Comp_p = Comp_cycles / num_mem_insts

    # ── Step 3: Parallelism metrics ──
    # CWP: Eq (8)-(9)
    CWP_full = (Mem_cycles + Comp_cycles) / Comp_cycles
    CWP = min(CWP_full, N)

    # MWP: Eq (5), (6), (16), (17)
    MWP_Without_BW = Mem_L / gpu.departure_delay

    # Eq (7): bandwidth consumed by one warp
    load_bytes_per_warp = gpu.warp_size * gpu.load_bytes_per_access
    BW_per_warp = clock_hz * load_bytes_per_warp / Mem_L

    # Eq (6): MWP limited by peak memory bandwidth
    num_active_sms = min(num_blocks, gpu.n_sms)
    mem_bw_bytes = gpu.mem_bw_gbps * 1e9
    MWP_peak_BW = mem_bw_bytes / (BW_per_warp * num_active_sms)

    # Eq (5): final MWP
    MWP = min(MWP_Without_BW, MWP_peak_BW, N)
    MWP = max(MWP, 1.0)

    # ── Step 4: Execution time (three cases) ──
    active_blocks_per_sm = max(1, N // warps_per_block)
    num_rep = num_blocks / (active_blocks_per_sm * num_active_sms)
    num_rep = max(num_rep, 1.0)

    if CWP >= MWP:
        if MWP >= N:
            # Case 1: Sufficient parallelism (Eq 22)
            exec_cycles = (Mem_cycles + Comp_cycles
                           + Comp_p * (MWP - 1)) * num_rep
            case = "sufficient"
        else:
            # Case 2: Memory-bound (Eq 23)
            exec_cycles = (Mem_cycles * (N / MWP)
                           + Comp_p * (MWP - 1)) * num_rep
            case = "memory_bound"
    else:
        # Case 3: Compute-bound (Eq 24)
        exec_cycles = (Mem_L + Comp_cycles * N) * num_rep
        case = "compute_bound"

    # ── Step 5: Convert to GFLOPS ──
    total_time_s = exec_cycles / clock_hz
    total_flops = num_blocks * threads_per_block * iterations * ai
    gflops = total_flops / total_time_s / 1e9 if total_time_s > 0 else 0.0

    return {
        "gflops": gflops,
        "case": case,
        "exec_cycles": exec_cycles,
        "time_ms": total_time_s * 1000,
        "MWP": MWP,
        "CWP": CWP,
        "Comp_cycles": Comp_cycles,
        "Mem_cycles": Mem_cycles,
        "Comp_p": Comp_p,
        "num_rep": num_rep,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Volkov Model (for comparison)
# ══════════════════════════════════════════════════════════════════════════

def predict_volkov(
    gpu: GPUConfig,
    ai: int,
    n_warps_per_sm: int,
    iterations: int,
    num_blocks: int,
    threads_per_block: int,
) -> Dict:
    """
    Predict GFLOPS using Volkov's throughput model (Little's Law).

    T_m = min(n/(mem_lat + AI×alu_lat),  mem_thru,  issue_thru/(AI+1))
    GFLOPS = 32 × AI × N_SM × f_SM × T_m
    """
    if ai == 0 or n_warps_per_sm == 0:
        return {"gflops": 0.0, "bound": "N/A"}

    clock_hz = gpu.clock_ghz * 1e9

    # mem_thru in IPC/SM
    # Each warp-instruction loads warp_size × load_bytes = 32 × 8 = 256 bytes
    bytes_per_mem_inst = gpu.warp_size * gpu.load_bytes_per_access  # 256
    bw_per_cycle = (gpu.mem_bw_gbps * 1e9) / clock_hz
    mem_thru = bw_per_cycle / (gpu.n_sms * bytes_per_mem_inst)

    # Three bounds
    latency_bound = n_warps_per_sm / (gpu.mem_lat_cycles + ai * gpu.alu_lat)
    memory_bound = mem_thru
    issue_bound = gpu.issue_thru / (ai + 1)

    T_m = min(latency_bound, memory_bound, issue_bound)

    # Identify binding constraint
    if T_m == latency_bound:
        bound = "latency"
    elif T_m == memory_bound:
        bound = "memory"
    else:
        bound = "issue"

    gflops = gpu.warp_size * ai * gpu.n_sms * gpu.clock_ghz * T_m

    return {"gflops": gflops, "bound": bound, "T_m": T_m}


# ══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gpu = GPU_CONFIGS["RTX5000"]
    print(f"=== {gpu.name} ===")
    print(f"{'AI':>4} {'HK GFLOPS':>10} {'HK Case':>16} "
          f"{'Volkov GFLOPS':>14} {'Volkov Bound':>13}")
    print("-" * 65)

    for ai in [2, 4, 8, 16, 32, 64, 128, 256]:
        hk = predict_hong_kim(gpu, ai, n_warps_per_sm=32,
                              iterations=500, num_blocks=1000,
                              threads_per_block=256)
        vk = predict_volkov(gpu, ai, n_warps_per_sm=32,
                            iterations=500, num_blocks=1000,
                            threads_per_block=256)
        print(f"{ai:>4} {hk['gflops']:>10.1f} {hk['case']:>16} "
              f"{vk['gflops']:>14.1f} {vk['bound']:>13}")
