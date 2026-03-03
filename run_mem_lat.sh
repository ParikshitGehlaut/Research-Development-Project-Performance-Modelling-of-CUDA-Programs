#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Memory Latency Measurement (Volkov's Methodology)
# Uses 4-byte (uint32_t) offset-based pointer chasing
#==============================================================================
SRC="src/mem_lat_kernel.cu"
EXE="./bin/mem_lat_kernel.out"
ARCH="sm_90"               # sm_75=Turing, sm_80=Ampere, sm_90=Hopper
OUT_DIR="Results/H100/MemLatency"
OUT_FILE="${OUT_DIR}/latency_summary.csv"

ARRAY_SIZE_MB=512
ITERATIONS=20000
NUM_BLOCKS=1000

SHMEM_KB_SWEEP=(0 4 8 16 32 48 64 72 80 96)
THREADS_PER_BLOCK_SWEEP=(32 64 128 256 512)

mkdir -p bin "${OUT_DIR}"

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

echo "ThreadsPerBlock,Shmem_KB,MeanLatency_cycles,MinLatency_cycles" > ${OUT_FILE}

echo "========================================================"
echo "Starting memory latency sweep (4-byte loads)..."
echo "Results will be saved to ${OUT_FILE}"

LOWEST_MEAN_LATENCY=999999
BEST_CONFIG=""

for TPB in "${THREADS_PER_BLOCK_SWEEP[@]}"; do
    for KB in "${SHMEM_KB_SWEEP[@]}"; do
        BYTES=$((KB * 1024))
        if [ ${BYTES} -eq 0 ]; then
            BYTES=4
        fi

        echo "--------------------------------------------------------"
        echo "Running: TPB=${TPB}, SHMEM=${KB}KB"

        CMD="${EXE} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"

        PROGRAM_OUT=$(${CMD} 2>&1) || {
            echo "Run failed for TPB=${TPB}, SHMEM=${KB}KB"
            echo "${TPB},${KB},FAIL,FAIL" >> ${OUT_FILE}
            continue
        }

        MEAN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MeanLatency_cycles: *\([0-9.]*\).*/\1/p')
        MIN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MinLatency_cycles: *\([0-9.]*\).*/\1/p')

        MEAN_LATENCY=${MEAN_LATENCY:-ERROR}
        MIN_LATENCY=${MIN_LATENCY:-ERROR}

        echo "${TPB},${KB},${MEAN_LATENCY},${MIN_LATENCY}" | tee -a ${OUT_FILE}

        if [[ $MEAN_LATENCY != "ERROR" && $MEAN_LATENCY != "0.00" ]]; then
            if (( $(echo "$MEAN_LATENCY < $LOWEST_MEAN_LATENCY" | bc -l) )); then
                LOWEST_MEAN_LATENCY=$MEAN_LATENCY
                BEST_CONFIG="TPB=${TPB}, SHMEM=${KB}KB"
            fi
        fi
    done
done

echo "========================================================"
echo "All sweeps completed."
echo
echo "Unloaded Global Memory Latency (Volkov's Method, 4-byte loads):"
echo "  Value (Minimum of Means): ${LOWEST_MEAN_LATENCY} cycles"
echo "  Found at Config: ${BEST_CONFIG}"
echo "========================================================"
