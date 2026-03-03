#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Synthetic Kernel with Streaming Memory (Volkov's Methodology)
# Uses 4-byte (uint32_t) loads — apple-to-apple comparison with Volkov
#==============================================================================
SRC="src/synthetic_kernel_streaming.cu"
EXE="./bin/synthetic_kernel_streaming.out"
ARCH="sm_90"               # sm_75=Turing, sm_80=Ampere, sm_90=Hopper

# Kernel execution settings
ITERATIONS=500
ARRAY_SIZE_MB=1024
NUM_BLOCKS=1000

# Sweep parameters
ARITH_INTENSITIES=(1 2 4 8 16 32 64 128 256 512)
SHMEM_KB=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96)
THREADS_PER_BLOCKS=(32 64 128 256)

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================
mkdir -p bin Results/H100/synthetic_streaming

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

#==============================================================================
# MAIN NESTED LOOP
#==============================================================================
for AI in "${ARITH_INTENSITIES[@]}"; do
    OUT="Results/H100/synthetic_streaming/results_a${AI}.csv"

    echo "THREADS_PER_BLOCK,SHMEM_KB,ExecutionTime_ms,GFLOPS,MaxAttainedOccupancy" > ${OUT}

    echo "========================================================"
    echo "Starting runs for Arithmetic Intensity = ${AI}"
    echo "Results will be saved to ${OUT}"

    for TPB in "${THREADS_PER_BLOCKS[@]}"; do
        echo "--------------------------------------------------------"
        echo "Threads per block: ${TPB}"

        for KB in "${SHMEM_KB[@]}"; do
            BYTES=$((KB * 1024))
            echo "Running: AI=${AI}, TPB=${TPB}, SHMEM=${KB}KB (${BYTES} bytes)"

            CMD="${EXE} ${AI} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"
            echo "CMD: ${CMD}"

            PROGRAM_OUT=$(${CMD} 2>&1) || {
                echo "Run failed for AI=${AI}, TPB=${TPB}, SHMEM=${KB}KB"
                echo "${PROGRAM_OUT}"
                echo "${TPB},${KB},FAIL,FAIL,FAIL" >> ${OUT}
                continue
            }

            EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
            GFLOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GFLOPS: *\([0-9.eE+-]*\).*/\1/p')
            OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

            EXEC_MS=${EXEC_MS:-ERROR}
            GFLOPS=${GFLOPS:-ERROR}
            OCC=${OCC:-ERROR}

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
echo "Results are stored in Results/H100/synthetic_streaming/"
