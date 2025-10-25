#!/usr/bin/env bash
set -euo pipefail

ARITH_INTENSITY=16

SRC="src/simplistic_kernel.cu"
mkdir -p bin
EXE="./bin/simplistic_kernel.out"
ARCH="sm_90"

OUT="Results/H100/simplistic/results_a${ARITH_INTENSITY}.csv"

ITERATIONS=2000000
ARRAY_SIZE_MB=512
THREADS_PER_BLOCK=256
NUM_BLOCKS=3000

SHMEM_KB=(0 4 8 12 16 20 24 28 32 36 40 44 48)

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

echo "SHMEM_KB,ExecutionTime_ms,Throughput_GOps,MaxAttainedOccupancy" > ${OUT}

echo "Starting runs for Arithmetic Intensity=${ARITH_INTENSITY}"
echo "Results will be in ${OUT}"

for KB in "${SHMEM_KB[@]}"; do
    BYTES=$((KB * 1024))
    echo "--------------------------------------------------------"
    echo "Running SHMEM=${KB}KB (${BYTES} bytes)"

    CMD="${EXE} ${ARITH_INTENSITY} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
    echo "CMD: ${CMD}"

    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ${KB}KB"
        echo "${PROGRAM_OUT}"
        echo "${KB},FAIL,FAIL,FAIL" >> ${OUT}
        continue
    }

    EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
    GOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GOps: *\([0-9.eE+-]*\).*/\1/p')
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    EXEC_MS=${EXEC_MS:-ERROR}
    GOPS=${GOPS:-ERROR}
    OCC=${OCC:-ERROR}

    echo "${KB},${EXEC_MS},${GOPS},${OCC}" | tee -a ${OUT}
done

echo "--------------------------------------------------------"
echo "Done. All results saved to ${OUT}"
