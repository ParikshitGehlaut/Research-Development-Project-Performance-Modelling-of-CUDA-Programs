#!/usr/bin/env bash
set -euo pipefail

# CONFIGURATION
SRC="src/tiled_matmul.cu"
EXE="./bin/tiled_matmul.out"
ARCH="sm_75" # sm_80 for A100, sm_90 for H100

N=4096
ITERATIONS=100
DUMMY_MB=1024
NUM_BLOCKS=1000

# Sweep Parameters
THREADS_PER_BLOCKS=(32 64 128 256)
SHMEM_KB_VALUES=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60)
TILE_SIZES=(8 16)

mkdir -p bin Results/RTX5000/matmul_tiled

# Compile
echo "Compiling ${SRC}..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compilation successful."

for T in "${TILE_SIZES[@]}"; do
    OUT="Results/RTX5000/matmul_tiled/results_tiled_n${N}_t${T}.csv"
    echo "THREADS_PER_BLOCK,TILE_DIM,SHMEM_KB,ExecutionTime_ms,GFLOPS,GBps,MaxAttainedOccupancy" > ${OUT}
    
    echo "=========================================="
    echo "Running Sweep for TILE_DIM = ${T}"
    echo "=========================================="

    for TPB in "${THREADS_PER_BLOCKS[@]}"; do
        echo "  > TPB: ${TPB}"
        
        for KB in "${SHMEM_KB_VALUES[@]}"; do
            BYTES=$((KB * 1024))
            
            CMD="${EXE} ${N} ${ITERATIONS} ${DUMMY_MB} ${BYTES} ${T} ${NUM_BLOCKS} ${TPB}"
            
            OUTPUT=$(${CMD} 2>&1) || { 
                echo "    ❌ Fail: TPB=${TPB} Shmem=${KB}KB"
                # Print the error output from C++ to understand why it failed
                echo "    Reason: $OUTPUT"
                echo "${TPB},${T},${KB},FAIL,FAIL,FAIL,FAIL" >> ${OUT}
                continue
            }

            # Parse
            MS=$(echo "$OUTPUT" | grep "ExecutionTime_ms" | awk '{print $2}')
            GFLOPS=$(echo "$OUTPUT" | grep "Throughput_GFLOPS" | awk '{print $2}')
            GBPS=$(echo "$OUTPUT" | grep "Throughput_GBps" | awk '{print $2}')
            OCC=$(echo "$OUTPUT" | grep "MaximumAttainedOccupancy" | awk '{print $2}')

            MS=${MS:-0}
            GFLOPS=${GFLOPS:-0}
            GBPS=${GBPS:-0}
            OCC=${OCC:-0}

            echo "    T=${T}, TPB=${TPB}, Shmem=${KB}KB -> Occ=${OCC}, GFLOPS=${GFLOPS}"
            echo "${TPB},${T},${KB},${MS},${GFLOPS},${GBPS},${OCC}" >> ${OUT}
        done
    done
    echo "✅ Saved results to ${OUT}"
done