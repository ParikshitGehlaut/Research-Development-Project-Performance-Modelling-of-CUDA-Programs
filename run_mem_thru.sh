#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# SCRIPT CONFIGURATION
#==============================================================================
# We MUST set Arith Intensity to 0 to isolate memory performance.
ARITH_INTENSITY=0
FIXED_SHMEM_BYTES=64

SRC="src/simplistic_kernel.cu"
EXE="./bin/simplistic_kernel.out"
ARCH="sm_75"               # Target GPU architecture

# Output CSV file name for this specific test
OUT="Results/RTX5000/mem_thru/unstructured_mem_thru.csv"

# Kernel execution settings
ITERATIONS=2000
THREADS_PER_BLOCK=256
# Launch many blocks to ensure the GPU is saturated
NUM_BLOCKS=3000

# --- ARRAY SIZE SWEEP (Outer Loop) ---
# List of array sizes (in MB) to test.
ARRAY_SIZES_MB=(1 2 4 8 16 32 64 128 256 512 1024)

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

# Compile the CUDA source file using nvcc.
echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

mkdir -p Results/RTX5000/mem_thru Results/H100/mem_thru Results/A100/mem_thru

# Create CSV header
echo "Array_Size_MB,Throughput_GBps,MaxAttainedOccupancy" > ${OUT}

echo "Starting runs for Unstructured Memory Throughput (A=${ARITH_INTENSITY})"
echo "Using fixed SHMEM = ${FIXED_SHMEM_BYTES} bytes"
echo "Results will be in ${OUT}"

# --- OUTER LOOP ---
for SIZE_MB in "${ARRAY_SIZES_MB[@]}"; do
    echo "========================================================"
    echo "RUNNING FOR ARRAY_SIZE_MB = ${SIZE_MB}"

    # Construct the command line arguments
    # Order: <exe> <arith_intensity> <iterations> <array_size_mb> <shmem_bytes> <threads_per_block> <num_blocks>
    CMD="${EXE} ${ARITH_INTENSITY} ${ITERATIONS} ${SIZE_MB} ${FIXED_SHMEM_BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
    echo "CMD: ${CMD}"

    # Execute the command. Capture stdout and stderr.
    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ${SIZE_MB}MB";
        echo "${PROGRAM_OUT}";
        echo "${SIZE_MB},FAIL,FAIL">>${OUT};
        continue;
    }

    # Extract results using sed
    GBPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GBps: *\([0-9.eE+-]*\).*/\1/p')
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    # Default values for error handling
    GBPS=${GBPS:-ERROR}
    OCC=${OCC:-ERROR}

    # Write results to CSV and console
    echo "${SIZE_MB},${GBPS},${OCC}" | tee -a ${OUT}
done

echo "========================================================"
echo "Done. All results saved to ${OUT}"