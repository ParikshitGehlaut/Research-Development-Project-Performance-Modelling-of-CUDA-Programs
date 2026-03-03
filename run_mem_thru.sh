#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Effective Memory Throughput Measurement (Volkov's Methodology)
# Uses 4-byte (uint32_t) streaming coalesced loads to saturate memory bus
# Fixed 512MB array — guarantees all accesses hit DRAM (bypasses L2 cache)
#==============================================================================
SRC="src/mem_thru_kernel.cu"
EXE="./bin/mem_thru_kernel.out"
ARCH="sm_90"               # sm_75=Turing, sm_80=Ampere, sm_90=Hopper

OUT="Results/H100/mem_thru/streaming_mem_thru.csv"

# Fixed large array to guarantee DRAM hits
# RTX 5000 L2 = 4MB, A100 L2 = 40MB, H100 L2 = 50MB
# 512MB >> all L2 caches → every access hits DRAM
ARRAY_SIZE_MB=512

# Launch config to saturate the bus
THREADS_PER_BLOCK=256
FIXED_SHMEM_BYTES=64

# Sweep iterations to vary total bytes transferred
# More iterations = more accurate timing (amortizes launch overhead)
ITERATION_SWEEP=(100 200 500 1000 2000)

mkdir -p bin Results/RTX5000/mem_thru Results/A100/mem_thru Results/H100/mem_thru

echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

echo "Iterations,Num_Blocks,Throughput_GBps,MaxAttainedOccupancy" > ${OUT}

echo "Starting Streaming Memory Throughput Measurement (4-byte loads)"
echo "Array Size: ${ARRAY_SIZE_MB} MB (fixed, >> L2 cache)"
echo "Results will be in ${OUT}"

NUM_ELEMENTS=$((ARRAY_SIZE_MB * 1024 * 1024 / 4))

for ITERS in "${ITERATION_SWEEP[@]}"; do
    # Calculate max blocks that fit: elements_needed = iters * tpb * blocks
    NEEDED_PER_BLOCK=$((ITERS * THREADS_PER_BLOCK))
    MAX_BLOCKS=$((NUM_ELEMENTS / NEEDED_PER_BLOCK))

    if [ ${MAX_BLOCKS} -eq 0 ]; then
        echo "Skipping ITERS=${ITERS}: too many iterations for ${ARRAY_SIZE_MB}MB array."
        continue
    fi

    # Cap at 1024 blocks (plenty for 48-114 SMs)
    if [ ${MAX_BLOCKS} -gt 1024 ]; then
        MAX_BLOCKS=1024
    fi

    echo "========================================================"
    echo "RUNNING: ITERS=${ITERS}, BLOCKS=${MAX_BLOCKS}"

    CMD="${EXE} ${ITERS} ${ARRAY_SIZE_MB} ${FIXED_SHMEM_BYTES} ${THREADS_PER_BLOCK} ${MAX_BLOCKS}"
    echo "CMD: ${CMD}"

    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ITERS=${ITERS}"
        echo "${PROGRAM_OUT}"
        echo "${ITERS},${MAX_BLOCKS},FAIL,FAIL" >> ${OUT}
        continue
    }

    GBPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GBps: *\([0-9.eE+-]*\).*/\1/p')
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    GBPS=${GBPS:-ERROR}
    OCC=${OCC:-ERROR}

    echo "${ITERS},${MAX_BLOCKS},${GBPS},${OCC}" | tee -a ${OUT}
done

echo "========================================================"
echo "Done. All results saved to ${OUT}"
echo "Use the highest Throughput_GBps value as your peak DRAM bandwidth."
