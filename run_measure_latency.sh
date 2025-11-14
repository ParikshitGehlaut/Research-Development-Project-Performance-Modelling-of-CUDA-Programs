#!/usr/bin/env bash
set -euo pipefail

SRC="src/measure_latency.cu"
EXE="./bin/measure_latency.out"
ARCH="sm_75" # sm_75=Turing, sm_80=Ampere, sm_90=Hopper
OUT_DIR="Results/RTX5000/Latency"
OUT_FILE="${OUT_DIR}/latency_summary.csv"

ARRAY_SIZE_MB=512
ITERATIONS=20000 
NUM_BLOCKS=1000 

SHMEM_KB_SWEEP=(0 4 8 16 32 48 64)
THREADS_PER_BLOCK_SWEEP=(32 64 128 256 512)

mkdir -p bin "${OUT_DIR}"

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

echo "ThreadsPerBlock,Shmem_KB,MeanLatency_cycles,MinLatency_cycles" > ${OUT_FILE}

echo "========================================================"
echo "Starting latency sweep..."
echo "Results will be saved to ${OUT_FILE}"

# --- MODIFIED TRACKING ---
# We track the lowest MEAN latency observed, not the lowest individual latency.
LOWEST_MEAN_LATENCY=999999
BEST_CONFIG=""

for TPB in "${THREADS_PER_BLOCK_SWEEP[@]}"; do
    for KB in "${SHMEM_KB_SWEEP[@]}"; do
        BYTES=$((KB * 1024))
        if [ ${BYTES} -eq 0 ]; then
            BYTES=4 # Minimal allocation if 0 requested
        fi

        echo "--------------------------------------------------------"
        echo "Running: TPB=${TPB}, SHMEM=${KB}KB"

        CMD="${EXE} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${TPB} ${NUM_BLOCKS}"
        # echo "CMD: ${CMD}"

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

        # --- MODIFIED LOGIC: Update the overall Minimum of Means ---
        if [[ $MEAN_LATENCY != "ERROR" && $MEAN_LATENCY != "0.00" ]]; then
            # Compare float values using bc
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
echo "Unloaded Global Memory Latency (Volkov's Method):"
echo "  Value (Minimum of Means): ${LOWEST_MEAN_LATENCY} cycles"
echo "  Found at Config: ${BEST_CONFIG}"
echo "========================================================"