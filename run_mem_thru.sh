#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Effective Memory Throughput Measurement (Volkov's Methodology)
# Uses 4-byte (uint32_t) streaming coalesced loads to saturate memory bus
#==============================================================================
SRC="src/mem_thru_kernel.cu"
EXE="./bin/mem_thru_kernel.out"
ARCH="sm_75"               # sm_75=Turing, sm_80=Ampere, sm_90=Hopper

OUT="Results/RTX5000/mem_thru/streaming_mem_thru.csv"

# Launch config to saturate the bus
ITERATIONS=2000
THREADS_PER_BLOCK=256
NUM_BLOCKS=3000
FIXED_SHMEM_BYTES=64       # Minimal shmem — maximize occupancy

# Array size sweep (MB)
ARRAY_SIZES_MB=(1 2 4 8 16 32 64 128 256 512 1024)

mkdir -p bin Results/RTX5000/mem_thru Results/A100/mem_thru Results/H100/mem_thru

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

echo "Array_Size_MB,Throughput_GBps,MaxAttainedOccupancy" > ${OUT}

echo "Starting Streaming Memory Throughput Measurement (4-byte loads)"
echo "Results will be in ${OUT}"

for SIZE_MB in "${ARRAY_SIZES_MB[@]}"; do
    echo "========================================================"
    echo "RUNNING FOR ARRAY_SIZE_MB = ${SIZE_MB}"

    CMD="${EXE} ${ITERATIONS} ${SIZE_MB} ${FIXED_SHMEM_BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
    echo "CMD: ${CMD}"

    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ${SIZE_MB}MB"
        echo "${PROGRAM_OUT}"
        echo "${SIZE_MB},FAIL,FAIL" >> ${OUT}
        continue
    }

    GBPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GBps: *\([0-9.eE+-]*\).*/\1/p')
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    GBPS=${GBPS:-ERROR}
    OCC=${OCC:-ERROR}

    echo "${SIZE_MB},${GBPS},${OCC}" | tee -a ${OUT}
done

echo "========================================================"
echo "Done. All results saved to ${OUT}"
