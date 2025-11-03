#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# SCRIPT CONFIGURATION
#==============================================================================
SRC="src/simplistic_kernel.cu"
EXE="./bin/simplistic_kernel.out"
ARCH="sm_90"               # Target GPU architecture (sm_75=Turing, sm_80=Ampere, sm_90=Hopper)

# Kernel execution settings
ITERATIONS=2000
ARRAY_SIZE_MB=1024
THREADS_PER_BLOCK=64
NUM_BLOCKS=1000

# Sweep parameters
ARITH_INTENSITIES=(0 1 2 4 8 16 32 64 128 256 512)
SHMEM_KB=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

mkdir -p bin Results/RTX5000/simplistic src \
         Results/H100/simplistic Results/A100/simplistic \
         Results/RTX5000/simplistic_with_streaming_memory \
         Results/H100/simplistic_with_streaming_memory \
         Results/A100/simplistic_with_streaming_memory

# Compile the CUDA source file using nvcc.
echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

#==============================================================================
# MAIN NESTED LOOP
#==============================================================================
for AI in "${ARITH_INTENSITIES[@]}"; do
    OUT="Results/RTX5000/simplistic/results_a${AI}.csv"

    # Create CSV header
    echo "SHMEM_KB,ExecutionTime_ms,GFLOPS,MaxAttainedOccupancy" > ${OUT}

    echo "========================================================"
    echo "Starting runs for Arithmetic Intensity = ${AI}"
    echo "Results will be saved to ${OUT}"

    # Sweep over shared memory sizes
    for KB in "${SHMEM_KB[@]}"; do
        BYTES=$((KB * 1024))
        echo "--------------------------------------------------------"
        echo "Running: AI=${AI}, SHMEM=${KB}KB (${BYTES} bytes)"

        # Construct command line
        CMD="${EXE} ${AI} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
        echo "CMD: ${CMD}"

        # Run and capture output
        PROGRAM_OUT=$(${CMD} 2>&1) || {
            echo "Run failed for AI=${AI}, SHMEM=${KB}KB"
            echo "${PROGRAM_OUT}"
            echo "${KB},FAIL,FAIL,FAIL" >> ${OUT}
            continue
        }

        # Extract metrics
        EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
        GFLOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GFLOPS: *\([0-9.eE+-]*\).*/\1/p')
        OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

        # Default values for error handling
        EXEC_MS=${EXEC_MS:-ERROR}
        GFLOPS=${GFLOPS:-ERROR}
        OCC=${OCC:-ERROR}

        # Write results to CSV and console
        echo "${KB},${EXEC_MS},${GFLOPS},${OCC}" | tee -a ${OUT}
    done

    echo "Completed runs for AI=${AI}"
    echo
done

echo "========================================================"
echo "All runs completed. Results saved under Results/RTX5000/simplistic/"
