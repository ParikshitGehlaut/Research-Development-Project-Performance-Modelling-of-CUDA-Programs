#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Arithmetic Latency Measurement (Volkov's Methodology)
# Sweeps FADD, FMUL, FMA across occupancy configurations
#==============================================================================
SRC="src/alu_lat_kernel.cu"
EXE="./bin/alu_lat_kernel.out"
ARCH="sm_75"               # sm_75=Turing, sm_80=Ampere, sm_90=Hopper
OUT_DIR="Results/RTX5000/ArithLatency"
OUT_FILE="${OUT_DIR}/arith_latency_summary.csv"

ITERATIONS=10000
NUM_BLOCKS=1000

# OpType: 1=FADD, 2=FMUL, 3=FMA
OP_SWEEP=(1 2 3)
OP_NAMES=("FADD" "FMUL" "FMA")
SHMEM_KB_SWEEP=(0 4 8 16 32 48 64)
THREADS_PER_BLOCK_SWEEP=(32 64 128 256)

mkdir -p bin "${OUT_DIR}"

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"
echo

echo "OpType,ThreadsPerBlock,Shmem_KB,MeanLatency_cycles,MinLatency_cycles" > ${OUT_FILE}

echo "========================================================"
echo "Starting arithmetic latency sweep..."
echo "Results will be saved to ${OUT_FILE}"

LOWEST_MEAN_LATENCY_FADD=999
LOWEST_MEAN_LATENCY_FMUL=999
LOWEST_MEAN_LATENCY_FMA=999

for OP_IDX in "${!OP_SWEEP[@]}"; do
    OP_TYPE=${OP_SWEEP[$OP_IDX]}
    OP_NAME=${OP_NAMES[$OP_IDX]}
    echo "********************************************************"
    echo "                Sweeping for ${OP_NAME} (Type ${OP_TYPE})"
    echo "********************************************************"

    LOWEST_MEAN_LATENCY_THIS_OP=999999
    BEST_CONFIG_THIS_OP=""

    for TPB in "${THREADS_PER_BLOCK_SWEEP[@]}"; do
        for KB in "${SHMEM_KB_SWEEP[@]}"; do
            BYTES=$((KB * 1024))
            if [ ${BYTES} -eq 0 ]; then
                BYTES=4
            fi

            echo "--------------------------------------------------------"
            echo "Running: Op=${OP_NAME}, TPB=${TPB}, SHMEM=${KB}KB"

            CMD="${EXE} ${OP_TYPE} ${ITERATIONS} ${BYTES} ${TPB} ${NUM_BLOCKS}"

            PROGRAM_OUT=$(${CMD} 2>&1) || {
                echo "Run failed for Op=${OP_NAME}, TPB=${TPB}, SHMEM=${KB}KB"
                echo "${OP_NAME},${TPB},${KB},FAIL,FAIL" >> ${OUT_FILE}
                continue
            }

            MEAN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MeanLatency_cycles: *\([0-9.]*\).*/\1/p')
            MIN_LATENCY=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MinLatency_cycles: *\([0-9.]*\).*/\1/p')

            MEAN_LATENCY=${MEAN_LATENCY:-ERROR}
            MIN_LATENCY=${MIN_LATENCY:-ERROR}

            echo "${OP_NAME},${TPB},${KB},${MEAN_LATENCY},${MIN_LATENCY}" | tee -a ${OUT_FILE}

            if [[ $MEAN_LATENCY != "ERROR" && $MEAN_LATENCY != "0.00" ]]; then
                if (( $(echo "$MEAN_LATENCY < $LOWEST_MEAN_LATENCY_THIS_OP" | bc -l) )); then
                    LOWEST_MEAN_LATENCY_THIS_OP=$MEAN_LATENCY
                    BEST_CONFIG_THIS_OP="TPB=${TPB}, SHMEM=${KB}KB"
                fi
            fi
        done
    done

    echo
    echo "Best result for ${OP_NAME}: ${LOWEST_MEAN_LATENCY_THIS_OP} cycles (@ ${BEST_CONFIG_THIS_OP})"
    echo

    if [ ${OP_TYPE} -eq 1 ]; then LOWEST_MEAN_LATENCY_FADD=$LOWEST_MEAN_LATENCY_THIS_OP; fi
    if [ ${OP_TYPE} -eq 2 ]; then LOWEST_MEAN_LATENCY_FMUL=$LOWEST_MEAN_LATENCY_THIS_OP; fi
    if [ ${OP_TYPE} -eq 3 ]; then LOWEST_MEAN_LATENCY_FMA=$LOWEST_MEAN_LATENCY_THIS_OP; fi
done

echo "========================================================"
echo "All sweeps completed."
echo
echo "Unloaded Arithmetic Latency (Volkov's Method):"
echo "  FADD (Minimum of Means): ${LOWEST_MEAN_LATENCY_FADD} cycles"
echo "  FMUL (Minimum of Means): ${LOWEST_MEAN_LATENCY_FMUL} cycles"
echo "  FMA  (Minimum of Means): ${LOWEST_MEAN_LATENCY_FMA}  cycles"
echo "========================================================"
