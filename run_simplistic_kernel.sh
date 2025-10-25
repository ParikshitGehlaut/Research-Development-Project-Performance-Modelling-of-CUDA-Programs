#!/usr/bin/env bash

set -euo pipefail

ARITH_INTENSITY=16       # Alpha
REGISTER_PRESSURE=0     # Controls dummy FMA operations. Set to 0 for baseline.
                        # Common values to test: 0, 1, 2, 5, 10

# Source and executable names
SRC="src/simplistic_kernel.cu" # Name of the CUDA C++ source file
mkdir -p bin
EXE="./bin/simplistic_kernel.out"  # Name of the compiled executable
ARCH="sm_75"               # Target GPU architecture (e.g., sm_75=Turing, sm_80/sm_86=Ampere, sm_90=Hopper)

# Output CSV file name includes arithmetic intensity ('a') and register pressure ('p').
OUT="Results/RTX5000/simplistic/results_a${ARITH_INTENSITY}_p${REGISTER_PRESSURE}.csv"

# Kernel execution settings
ITERATIONS=2000000      # Number of pointer-chasing steps per thread. Adjust for reasonable runtime.
ARRAY_SIZE_MB=512       # Size of the global memory array for pointer chasing (in MB).
THREADS_PER_BLOCK=256   # Number of threads per CUDA block. Must be multiple of warpSize (usually 32).
NUM_BLOCKS=300          # Number of CUDA blocks to launch. Total threads = NUM_BLOCKS * THREADS_PER_BLOCK.


SHMEM_KB=(0 2 4 6 8 12 16 20 24 28 32 36 40 44 48) # Example range, adjust based on GPU limits

#==============================================================================
# SCRIPT EXECUTION
#==============================================================================

# Compile the CUDA source file using nvcc.
echo "Compiling..."
# -O3: Enable optimizations.
# -arch: Specify the target GPU architecture.
# ${SRC}: Input source file.
# -o ${EXE}: Output executable file name.
nvcc -O3 -arch=${ARCH} ${SRC} -o ${EXE}
echo "Compiled ${EXE}"

# Write the header row to the CSV output file. Overwrites if file exists.
echo "SHMEM_KB,RegisterPressure,ExecutionTime_ms,Throughput_GOps,MaxAttainedOccupancy" > ${OUT}

echo "Starting runs for Arithmetic Intensity=${ARITH_INTENSITY}, Register Pressure=${REGISTER_PRESSURE}."
echo "Results will be in ${OUT}"

# Loop through each shared memory size defined in the SHMEM_KB array.
for KB in "${SHMEM_KB[@]}"; do
    # Convert KB to Bytes.
    BYTES=$((KB * 1024))
    echo "--------------------------------------------------------"
    echo "Running SHMEM=${KB}KB (${BYTES} bytes)"

    # Construct the command line arguments for the executable.
    # Order matches the expectations in the C++ main function:
    # <exe> <arith_intensity> <register_pressure> <iterations> <array_size_mb> <shmem_bytes> <threads_per_block> <num_blocks>
    CMD="${EXE} ${ARITH_INTENSITY} ${REGISTER_PRESSURE} ${ITERATIONS} ${ARRAY_SIZE_MB} ${BYTES} ${THREADS_PER_BLOCK} ${NUM_BLOCKS}"
    echo "CMD: ${CMD}"

    # Execute the command. Capture stdout and stderr into PROGRAM_OUT variable.
    # If the command fails (||), print error, log failure to CSV, and continue to next loop iteration.
    PROGRAM_OUT=$(${CMD} 2>&1) || {
        echo "Run failed for ${KB}KB";
        echo "${PROGRAM_OUT}";
        # Append a failure entry to the CSV file.
        echo "${KB},${REGISTER_PRESSURE},FAIL,FAIL,FAIL">>${OUT};
        continue; # Skip to the next shared memory size
    }

    # Parse the output from the C++ program to extract the results.
    # Uses `sed` with capture groups to find lines matching specific patterns and extract the numeric values.
    # -n: Suppress default output.
    # s/.../\1/p: Substitute the matched line with the captured group (\1) and print (p).
    REG_PRES=$(echo "${PROGRAM_OUT}" | sed -n 's/.*RegisterPressure: *\([0-9]*\).*/\1/p')
    EXEC_MS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*ExecutionTime_ms: *\([0-9.]*\).*/\1/p')
    GOPS=$(echo "${PROGRAM_OUT}" | sed -n 's/.*Throughput_GOps: *\([0-9.eE+-]*\).*/\1/p') # Allows scientific notation
    OCC=$(echo "${PROGRAM_OUT}" | sed -n 's/.*MaximumAttainedOccupancy_warpsPerSM: *\([0-9]*\).*/\1/p')

    # Provide default values in case parsing failed (e.g., output format changed).
    # Uses bash parameter expansion: ${VAR:-DEFAULT} uses DEFAULT if VAR is unset or null.
    REG_PRES=${REG_PRES:-${REGISTER_PRESSURE}} # Fallback to the script's intended pressure value
    EXEC_MS=${EXEC_MS:-ERROR}                  # Indicate error if parsing fails
    GOPS=${GOPS:-ERROR}
    OCC=${OCC:-ERROR}

    # Write the extracted values to the CSV file.
    # `tee -a` appends the output to the file *and* prints it to the console.
    echo "${KB},${REG_PRES},${EXEC_MS},${GOPS},${OCC}" | tee -a ${OUT}

done # End of loop through shared memory sizes

echo "--------------------------------------------------------"
echo "Done. All results saved to ${OUT}"