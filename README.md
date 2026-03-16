# Analytical GPU Performance Modeling

This repository contains the implementation artifacts for an analytical GPU performance modeling study. The goal is to predict GPU throughput (GFLOP/s) as a function of occupancy (active warps per SM) using Volkov's model, and to motivate that choice by comparing against two earlier models: Hong & Kim (ISCA 2009) and Zhang & Owens (HPCA 2011).

---

## Repository Structure

```
.
├── src/                          # CUDA microkernels
│   ├── alu_lat_kernel.cu         # ALU latency microkernel (FADD/FMUL/FMA)
│   ├── mem_lat_kernel.cu         # Global memory latency microkernel
│   ├── mem_thru_kernel.cu        # Global memory throughput microkernel
│   └── synthetic_kernel_streaming.cu  # Synthetic kernel for GFLOP/s vs occupancy
├── motivation/                   # Analytical models and comparison plots
│   ├── hong_kim_model_fixed.py   # Hong & Kim 2009 model (corrected)
│   ├── zhang_owens_model.py      # Zhang & Owens 2011 model
│   ├── compare_models_plot_fixed.py   # H&K vs Volkov comparison plots
│   ├── compare_zo_volkov_plot.py      # Z&O vs Volkov comparison plots
│   ├── A100/                     # Per-GPU model output plots
│   ├── H100/
│   └── RTX5000/
├── Results/                      # Measured CSV data
│   ├── A100/
│   ├── H100/
│   └── RTX5000/
├── Images/                       # Merged result images
├── run_alu_lat.sh                # Script to run ALU latency sweep
├── run_mem_lat.sh                # Script to run memory latency sweep
├── run_mem_thru.sh               # Script to run memory throughput sweep
└── run_synthetic.sh              # Script to run synthetic kernel sweep
```

---

## Microkernels (`src/`)

All three microkernels follow Volkov's methodology: they control occupancy by varying the amount of shared memory allocated per block (which limits how many blocks can reside on each SM simultaneously) while measuring the quantity of interest via hardware clock counters.

### 1. ALU Latency (`alu_lat_kernel.cu`)

Measures the unloaded latency of FP32 arithmetic instructions (FADD, FMUL, FMA) using dependent instruction chains. Each kernel creates a long register-to-register dependency chain so that every instruction must wait for the result of the previous one, directly exposing the pipeline latency.

- **Operation types**: FADD, FMUL, FMA (selected via command-line argument)
- **Access pattern**: Register-to-register — no global or shared memory reads
- **Output**: Mean and minimum latency in GPU clock cycles per operation
- **Usage** (via shell script):
  ```bash
  bash run_alu_lat.sh
  ```

### 2. Memory Latency (`mem_lat_kernel.cu`)

Measures the unloaded global memory access latency using a pointer-chasing (dependent load) chain. Each load depends on the result of the previous load, preventing any out-of-order or prefetch speculation and thus exposing the true round-trip DRAM latency.

- **Access pattern**: Coalesced, streaming, dependent 4-byte (`uint32_t`) loads
- **Each warp load**: 32 threads × 4 bytes = 128 bytes
- **Output**: Mean and minimum latency in GPU clock cycles per load
- **Usage** (via shell script):
  ```bash
  bash run_mem_lat.sh
  ```

### 3. Memory Throughput (`mem_thru_kernel.cu`)

Measures the effective peak global memory bandwidth by saturating the memory bus with coalesced, streaming loads. A large array (512 MB, well above the L2 cache capacity) is used to ensure every access reaches DRAM.

- **Access pattern**: Streaming, coalesced, non-repeating 4-byte loads (AI = 0)
- **Output**: Throughput in GB/s and maximum attained occupancy (warps/SM)
- **Usage** (via shell script):
  ```bash
  bash run_mem_thru.sh
  ```

---

## Measuring GFLOP/s vs Occupancy (`src/synthetic_kernel_streaming.cu`)

The synthetic kernel implements Volkov's parameterized workload to collect measured GFLOP/s across the full occupancy range. Each kernel iteration performs:

1. One 4-byte dependent global memory load
2. A configurable number of dependent FMUL operations (the *arithmetic intensity*, AI)

By sweeping both the shared memory per block (to vary occupancy) and the arithmetic intensity, the script produces a dataset of (warps/SM, GFLOP/s) pairs for each AI value. These measured curves are later compared against the analytical models.

- **Supported AI values**: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- **Output**: CSV files with columns `THREADS_PER_BLOCK`, `SHMEM_KB`, `ExecutionTime_ms`, `GFLOPS`, `MaxAttainedOccupancy`
- **Usage** (via shell script):
  ```bash
  bash run_synthetic.sh
  ```

---

## Analytical Models (`motivation/`)

The scripts in `motivation/` implement three analytical models and generate comparison plots of predicted vs. measured throughput as a function of occupancy (warps/SM).

### Volkov's Model (primary)

Volkov's model is a closed-form application of **Little's Law** to the GPU execution pipeline. For a kernel with arithmetic intensity AI and occupancy *n* warps/SM, the throughput is:

```
T(n) = warp_size × AI × min(
    n / (mem_lat + AI × alu_lat),   # latency-limited regime
    mem_throughput / warp_load,      # memory bandwidth limited
    issue_throughput / (AI + 1)      # issue slot limited
)
```

The three terms correspond to the three possible bottlenecks: latency hiding (linear growth with occupancy), memory bandwidth saturation, and instruction issue throughput saturation. This gives the characteristic shape of a curve that rises linearly at low occupancy and then plateaus.

Volkov's model is preferred because it:
- Correctly captures the shape of measured GFLOP/s vs. occupancy curves across all tested GPUs (RTX 5000, A100, H100)
- Has a simple closed form requiring only three micro-benchmarked parameters: `alu_lat`, `mem_lat`, and peak memory bandwidth
- Naturally handles all three execution regimes (latency-bound, memory-bound, issue-bound) with a single `min()` expression

### Hong & Kim 2009 (`hong_kim_model_fixed.py`)

The Hong & Kim model (ISCA 2009) derives GPU execution time from two key quantities, the *Memory Warp Parallelism* (MWP) and *Compute Warp Parallelism* (CWP), and uses them to identify which resource is the bottleneck. The model distinguishes three cases:

- **Sufficient parallelism** (MWP ≥ *n*): memory and compute fully overlap
- **Memory-bound** (CWP ≥ MWP): execution time grows with *n* / MWP
- **Compute-bound** (CWP < MWP): execution time grows with *n*

The implementation in `hong_kim_model_fixed.py` corrects a common mistake in applying the model to per-SM throughput: computing throughput directly for *n* warps on a single SM (without the `num_rep` scheduling multiplier that algebraically cancels and produces a flat curve). The script `compare_models_plot_fixed.py` generates side-by-side plots of H&K predictions vs. Volkov predictions alongside measured data, illustrating where H&K diverges from reality.

### Zhang & Owens 2011 (`zhang_owens_model.py`)

The Zhang & Owens model (HPCA 2011) characterizes the GPU with two occupancy-dependent throughput curves obtained from microkernels:

- `alu(n)` — throughput of a pure FP32 kernel (adds/cycle/SM)
- `mem(n)` — throughput of a pure memory kernel (warp-loads/cycle/SM)

For a mixed kernel with arithmetic intensity AI, the model gives:

```
Throughput = min(alu(n), AI × mem(n))   # full compute–memory overlap
```

Both `alu(n)` and `mem(n)` are themselves piecewise functions of *n* (latency-limited at low occupancy, hardware-peak-limited at high occupancy). The script `compare_zo_volkov_plot.py` generates side-by-side plots comparing Z&O predictions against Volkov's model and measured data, showing that Z&O and Volkov agree closely when the microbenchmark parameters are consistent, but Volkov's simpler formulation is easier to calibrate and apply.

---

## Running the Full Pipeline

### Prerequisites

- CUDA toolkit (nvcc) with architecture flag matching your GPU:
  - `sm_75` — Turing (e.g., RTX 5000)
  - `sm_80` — Ampere (e.g., A100)
  - `sm_90` — Hopper (e.g., H100)
- Python 3 with `numpy`, `pandas`, `matplotlib`

### Step 1: Measure hardware parameters

```bash
bash run_alu_lat.sh    # → Results/<GPU>/ArithLatency/arith_latency_summary.csv
bash run_mem_lat.sh    # → Results/<GPU>/MemLatency/latency_summary.csv
bash run_mem_thru.sh   # → Results/<GPU>/mem_thru/streaming_mem_thru.csv
```

Before running, set `ARCH` and `OUT_DIR` in each script to match your GPU.

### Step 2: Collect GFLOP/s vs occupancy data

```bash
bash run_synthetic.sh  # → Results/<GPU>/synthetic_streaming/results_a<AI>.csv
```

### Step 3: Generate model comparison plots

```bash
cd motivation

# Hong & Kim vs Volkov
python compare_models_plot_fixed.py --gpu RTX5000 --all
python compare_models_plot_fixed.py --gpu A100 --all
python compare_models_plot_fixed.py --gpu H100 --all

# Zhang & Owens vs Volkov
python compare_zo_volkov_plot.py --gpu RTX5000 --all
python compare_zo_volkov_plot.py --gpu A100 --all
python compare_zo_volkov_plot.py --gpu H100 --all
```

Plots are saved under `motivation/<GPU>/`.

---

## GPUs Tested

| GPU | Architecture | SMs | FP32 Cores/SM | Clock (GHz) | Mem BW (GB/s) | Mem Lat (cycles) |
|-----|-------------|-----|---------------|-------------|---------------|------------------|
| Quadro RTX 5000 | Turing | 48 | 64 | 1.815 | 340 | 236 |
| A100 | Ampere | 108 | 64 | 1.410 | 812 | 244 |
| H100 | Hopper | 114 | 128 | 1.755 | 1083 | 347 |

---

## References

- V. Volkov, "Understanding Latency Hiding on GPUs," Ph.D. Thesis, UC Berkeley, 2016.
- S. Hong and H. Kim, "An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness," ISCA 2009.
- Y. Zhang and J. D. Owens, "A Quantitative Performance Analysis Model for GPU Architectures," HPCA 2011.
