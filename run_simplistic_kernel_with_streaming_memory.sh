#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# SCRIPT CONFIGURATION
#==============================================================================
SRC="src/simplistic_kernel_with_streaming_memory.cu"
EXE="./bin/simplistic_kernel_with_streaming_memory.out"
ARCH="sm_75"               # Target GPU architecture (sm_75=Turing, sm_80=Ampere, sm_90=Hopper)

# Kernel execution settings
ITERATIONS=500
ARRAY_SIZE_MB=1024
NUM_BLOCKS=1000

# Sweep parameters
ARITH_INTENSITIES=(1 2 4 8 16 32 64 128 256 512)
SHMEM_KB=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)
THREADS_PER_BLOCKS=(32 64 128 256)

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================
# Create directories if they don't exist
mkdir -p bin Results/H100/simplistic_with_streaming_memory \
         Results/A100/simplistic_with_streaming_memory \
         Results/RTX5000/simplistic_with_streaming_memory

# Compile CUDA source
echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

#==============================================================================
# MAIN NESTED LOOP
#==============================================================================
for AI in "${ARITH_INTENSITIES[@]}"; do
    # Output file per arithmetic intensity
    OUT="Results/RTX5000/simplistic_with_streaming_memory/results_a${AI}.csv"

    # Create CSV header (include threads per block column)
    echo "THREADS_PER_BLOCK,SHMEM_KB,ExecutionTime_ms,GFLOPS,MaxAttainedOccupancy" > ${OUT}

    echo "========================================================"
    echo "Starting runs for Arithmetic Intensity = ${AI}"
    echo "Results will be saved to ${OUT}"

    # Loop over threads per block
    for TPB in "${THREADS_PER_BLOCKS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Threads per block: ${TPB}"

        # Loop over shared memory sizes
        for KB in "${SHMEM_KB[@]}"; do
            BYTES=$((KB * 1024))
            echo "Running: AI=${AI}, TPB=${TPB}, SHMEM=${KB}KB (${BYTES} bytes)"

            CMD="${EXE} ${AI} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"
            echo "CMD: ${CMD}"

            # Run and capture output
            PROGRAM_OUT=$(${CMD} 2>&1) || {
                echo "Run failed for AI=${AI}, TPB=${TPB}, SHMEM=${KB}KB"
                echo "${PROGRAM_OUT}"
                echo "${TPB},${KB},FAIL,FAIL,FAIL" >> ${OUT}
                continue
            }

            # Parse key results
            EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
            GFLOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GFLOPS: *\([0-9.eE+-]*\).*/\1/p')
            OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

            # Default values for error handling
            EXEC_MS=${EXEC_MS:-ERROR}
            GFLOPS=${GFLOPS:-ERROR}
            OCC=${OCC:-ERROR}

            # Append results to CSV
            echo "${TPB},${KB},${EXEC_MS},${GFLOPS},${OCC}" | tee -a ${OUT}
        done
        echo "Completed runs for AI=${AI}, TPB=${TPB}"
        echo
    done
    echo "Completed all TPB runs for AI=${AI}"
    echo
done

echo "========================================================"
echo "All sweeps completed successfully."
echo "Results are stored in Results/RTX5000/simplistic_with_streaming_memory/"
