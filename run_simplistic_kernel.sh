#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# SCRIPT CONFIGURATION
#==============================================================================
ARITH_INTENSITY=8

SRC="src/simplistic_kernel.cu"
EXE="./bin/simplistic_kernel.out"
ARCH="sm_75"               # Target GPU architecture (sm_75=Turing, sm_80=Ampere, sm_90=Hopper)

# Output CSV file name (Updated)
OUT="Results/RTX5000/simplistic/results_a${ARITH_INTENSITY}.csv"

# Kernel execution settings
ITERATIONS=2000
ARRAY_SIZE_MB=512
THREADS_PER_BLOCK=64
NUM_BLOCKS=3000

# --- OCCUPANCY SWEEP ---
# Array of shared memory sizes (in KB) to request per block for each run.
SHMEM_KB=(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64)

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================
# Create directories if they don't exist
mkdir -p bin Results/RTX5000/simplistic src Results/H100/simplistic Results/A100/simplistic
# Assuming your source file is now named simplistic_kernel_no_rp.cu and placed in src/
# If not, adjust the SRC variable and copy the file accordingly.
# cp path/to/your/simplistic_kernel_no_rp.cu src/

# Compile the CUDA source file using nvcc.
echo "Compiling..."
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

# Create CSV header
echo "SHMEM_KB,ExecutionTime_ms,GFLOPS,MaxAttainedOccupancy" > ${OUT}

echo "Starting runs for Arithmetic Intensity=${ARITH_INTENSITY}"
echo "Results will be in ${OUT}"

# Loop through each shared memory size defined in the SHMEM_KB array.
for KB in "${SHMEM_KB[@]}"; do
    # Convert KB to Bytes.
    BYTES=$((KB * 1024))
    echo "--------------------------------------------------------"
    echo "Running SHMEM=${KB}KB (${BYTES} bytes)"

    # Construct the command line arguments for the executable (Updated).
    # Order: <exe> <arith_intensity> <iterations> <array_size_mb> <shmem_bytes> <threads_per_block> <num_blocks>
    CMD="${EXE} ${ARITH_INTENSITY} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
    echo "CMD: ${CMD}"

    # Execute the command. Capture stdout and stderr into PROGRAM_OUT variable.
    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ${KB}KB";
        echo "${PROGRAM_OUT}";
        echo "${KB},FAIL,FAIL,FAIL">>${OUT};
        continue;
    }

    # Extract results using sed (same parsing logic)
    EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
    GFLOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GFLOPS: *\([0-9.eE+-]*\).*/\1/p')
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    # Default values for error handling
    EXEC_MS=${EXEC_MS:-ERROR}
    GFLOPS=${GFLOPS:-ERROR}
    OCC=${OCC:-ERROR}

    # Write results to CSV and console
    echo "${KB},${EXEC_MS},${GFLOPS},${OCC}" | tee -a ${OUT}
done

echo "--------------------------------------------------------"
echo "Done. All results saved to ${OUT}"