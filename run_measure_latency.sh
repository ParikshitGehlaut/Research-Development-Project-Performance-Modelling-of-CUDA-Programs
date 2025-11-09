#!/usr/bin/env bash
set -euo pipefail

SRC="src/measure_latency.cu"
EXE="./bin/measure_latency.out"
ARCH="sm_80" # sm_75=Turing, sm_80=Ampere, sm_90=Hopper
OUT_DIR="Results/A100/Latency"
OUT_FILE="${OUT_DIR}/latency_summary.csv"

ARRAY_SIZE_MB=512
ITERATIONS=20000 
NUM_BLOCKS=1000 

SHMEM_KB_SWEEP=(0 4 8 16 32 48 64 80 96)
THREADS_PER_BLOCK_SWEEP=(32 64 128 256 512)


mkdir -p bin "${OUT_DIR}"

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

# --- MODIFIED HEADER ---
echo "ThreadsPerBlock,Shmem_KB,MeanLatency_cycles,MinLatency_cycles" > ${OUT_FILE}

echo "========================================================"
echo "Starting latency sweep..."
echo "Results will be saved to ${OUT_FILE}"

# Initialize a variable to track the overall minimum latency
OVERALL_MIN_LATENCY=999999

for TPB in "${THREADS_PER_BLOCK_SWEEP[@]}"; do
    for KB in "${SHMEM_KB_SWEEP[@]}"; do
        BYTES=$((KB * 1024))
        if [ ${BYTES} -eq 0 ]; then
            BYTES=4
        fi

        echo "--------------------------------------------------------"
        echo "Running: TPB=${TPB}, SHMEM=${KB}KB"

        CMD="${EXE} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"
        echo "CMD: ${CMD}"

        PROGRAM_OUT=$(${CMD} 2>&1) || {
            echo "Run failed for TPB=${TPB}, SHMEM=${KB}KB"
            echo "${PROGRAM_OUT}"
            echo "${TPB},${KB},FAIL,FAIL" >> ${OUT_FILE}
            continue
        }

        # --- MODIFIED PARSING ---
        MEAN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MeanLatency_cycles: *\([0-9.]*\).*/\1/p')
        MIN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MinLatency_cycles: *\([0-9.]*\).*/\1/p')

        MEAN_LATENCY=${MEAN_LATENCY:-ERROR}
        MIN_LATENCY=${MIN_LATENCY:-ERROR}

        echo "${TPB},${KB},${MEAN_LATENCY},${MIN_LATENCY}" | tee -a ${OUT_FILE}

        # Update the overall minimum
        if [[ $MIN_LATENCY != "ERROR" && $MIN_LATENCY != "0.00" ]]; then
            if (( $(echo "$MIN_LATENCY < $OVERALL_MIN_LATENCY" | bc -l) )); then
                OVERALL_MIN_LATENCY=$MIN_LATENCY
            fi
        fi
    done
done

echo "========================================================"
echo "All sweeps completed."
echo
echo "Unloaded Global Memory Latency (Streaming): ${OVERALL_MIN_LATENCY} cycles"
echo "========================================================"