#!/usr/bin/env bash
set -euo pipefail
#==============================================================================
# SCRIPT CONFIGURATION
#==============================================================================
SRC="src/naive_matmul.cu"
EXE="./bin/naive_matmul.out"
ARCH="sm_90" # Target GPU architecture (sm_75=Turing, sm_80=Ampere, sm_90=Hopper)

# Matrix size sweep (repurposed from ARITH_INTENSITIES for comparison)
N_VALUES=(4096)
# Fixed parameters (dummies for CLI compat; ignored in code)
ITERATIONS=500
ARRAY_SIZE_MB=1024
NUM_BLOCKS=1000
# Other sweeps
SHMEM_KB=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)
THREADS_PER_BLOCKS=(32 64 128 256)
#==============================================================================
# SCRIPT EXECUTION
#==============================================================================
# Create directories if they don't exist
mkdir -p bin Results/H100/matmul_naive \
Results/A100/matmul_naive \
Results/RTX5000/matmul_naive

# Compile CUDA source
echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

#==============================================================================
# MAIN NESTED LOOP
#==============================================================================
for N in "${N_VALUES[@]}"; do
    # Output file per matrix size
    OUT="Results/H100/matmul_naive/results_n${N}.csv"
    # Create CSV header (include threads per block column; added GBps for completeness)
    echo "THREADS_PER_BLOCK,SHMEM_KB,ExecutionTime_ms,GFLOPS,GBps,MaxAttainedOccupancy" > ${OUT}
    echo "========================================================"
    echo "Starting runs for Matrix Size N = ${N}"
    echo "Results will be saved to ${OUT}"
    # Loop over threads per block
    for TPB in "${THREADS_PER_BLOCKS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Threads per block: ${TPB}"
        # Loop over shared memory sizes
        for KB in "${SHMEM_KB[@]}"; do
            BYTES=$((KB * 1024))
            echo "Running: N=${N}, TPB=${TPB}, SHMEM=${KB}KB (${BYTES} bytes)"
            CMD="${EXE} ${N} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"
            echo "CMD: ${CMD}"
            # Run and capture output
            PROGRAM_OUT=$(${CMD} 2>&1) || {
                echo "Run failed for N=${N}, TPB=${TPB}, SHMEM=${KB}KB"
                echo "${PROGRAM_OUT}"
                echo "${TPB},${KB},FAIL,FAIL,FAIL,FAIL" >> ${OUT}
                continue
            }
            # Parse key results
            EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
            GFLOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GFLOPS: *\([0-9.eE+-]*\).*/\1/p')
            GBPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GBps: *\([0-9.eE+-]*\).*/\1/p')
            OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')
            # Default values for error handling
            EXEC_MS=${EXEC_MS:-ERROR}
            GFLOPS=${GFLOPS:-ERROR}
            GBPS=${GBPS:-ERROR}
            OCC=${OCC:-ERROR}
            # Append results to CSV
            echo "${TPB},${KB},${EXEC_MS},${GFLOPS},${GBPS},${OCC}" | tee -a ${OUT}
        done
        echo "Completed runs for N=${N}, TPB=${TPB}"
        echo
    done
    echo "Completed all TPB runs for N=${N}"
    echo
done
echo "========================================================"
echo "All sweeps completed successfully."
echo "Results are stored in Results/H100/matmul_naive/"