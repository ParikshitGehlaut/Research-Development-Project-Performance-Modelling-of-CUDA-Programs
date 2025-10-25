#include <cuda.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <random>
#include <cstdint>
#include <cassert>
#include <numeric>

using namespace std;

#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(_e) << std::endl; \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// device function to get ID of SM on which current thread is running
static __device__ __forceinline__ unsigned smid(){
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
} 

// Structure to store the start clock, end clock, and SM ID for a single warp
struct WarpEvent {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
};

// type punning
union PtrDoubleUnion {
    uintptr_t ptr; // Memory address representation
    double    d;   // Floating-point representation of the same bits
};

template<int ARITH_INTENSITY, int REGISTER_PRESSURE> __global__ void simplistic_kernel(const uintptr_t* ptr_array,
                                                                                    const uintptr_t* start_addrs,
                                                                                    int iterations, 
                                                                                    WarpEvent* warp_events)
{
    extern __shared__ int shmen[];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x; // unique global ID of this thread
    
    // warpSize = 32 threads
    int warps_per_block = blockDim.x / warpSize; // number of warps per block
    int warp_id_in_block = threadIdx.x / warpSize; // ID of warp within the block
    int lane_id = threadIdx.x % warpSize; // ID of thread within its warps
    
    // Like global_thread_id, (block linear id) * (number of warps in block) + (warp in the block)
    int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;


    if (lane_id == 0) {
        warp_events[global_warp_id].start_clock = clock(); // Record start time 
        warp_events[global_warp_id].sm_id = smid();        // Record the SM this warp is running on
    }

    // ------------    Core Kernel Logic        -------------------
    uintptr_t start_ptr = start_addrs[global_thread_id];
    PtrDoubleUnion p_union;
    p_union.ptr = start_ptr;

    volatile double p_as_double = p_union.d; // a
    const double zero = 0.0; // b

    // Register Pressure Variable
    double p0 = threadIdx.x * 0.1, p1 = threadIdx.x * 0.2;
    double p2 = threadIdx.x * 0.3, p3 = threadIdx.x * 0.4;
    double p4 = threadIdx.x * 0.5, p5 = threadIdx.x * 0.6;
    double p6 = threadIdx.x * 0.7, p7 = threadIdx.x * 0.8;
    // Volatile sink to ensure the dummy calculations below aren't optimized out.
    volatile double dummy_sink = 0.0;


    for(int i=0; i<iterations; i++){
        // one memory load operation i.e a = memory[a]
        p_union.d = p_as_double; // reading GPU memory address as double
        uintptr_t next_ptr = *reinterpret_cast<const uintptr_t*>(p_union.ptr); // dereferencing it
        p_union.ptr = next_ptr;

        // Alpha (AI) arithmetic instructions i.e a = a + b
        #pragma unroll // Hint to the compiler to unroll this small loop.
        for (int j = 0; j < ARITH_INTENSITY; ++j) {
            p_as_double = p_as_double + zero;
        }

        // Operations to increase register pressure
        #pragma unroll
        for (int k = 0; k < REGISTER_PRESSURE; ++k) {
            p0 = p0 * p1 + p2;
            p1 = p1 * p2 + p3;
            p2 = p2 * p3 + p4;
            p3 = p3 * p4 + p5;
            p4 = p4 * p5 + p6;
            p5 = p5 * p6 + p7;
            p6 = p6 * p7 + p0;
            p7 = p7 * p0 + p1;
        }
        // from optimizing away the entire dummy calculation loop.
        dummy_sink = p7;
    }

    if (lane_id == 0) {
        __threadfence_system();
        warp_events[global_warp_id].end_clock = clock(); // Record end time
    }
}

template<int ARITH_INTENSITY> void launch_with_pressure(int register_pressure, dim3 gridDim, dim3 blockDim, size_t shmem_bytes,
                          const uintptr_t* d_ptr_array, const uintptr_t* d_start_addrs,
                          int iterations, WarpEvent* d_warp_events)
{
    switch (register_pressure) {
        case 0:  simplistic_kernel<ARITH_INTENSITY, 0><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 1:  simplistic_kernel<ARITH_INTENSITY, 1><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 2:  simplistic_kernel<ARITH_INTENSITY, 2><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 5:  simplistic_kernel<ARITH_INTENSITY, 5><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 10: simplistic_kernel<ARITH_INTENSITY, 10><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        default:
            cerr << "Unsupported register pressure: " << register_pressure << ". Please add a case for it." << endl;
            exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_kernel(int arith_intensity, int register_pressure, dim3 gridDim, dim3 blockDim, size_t shmem_bytes,
                   const uintptr_t* d_ptr_array, const uintptr_t* d_start_addrs,
                   int iterations, WarpEvent* d_warp_events)
{
    switch (arith_intensity) {
        case 0:   launch_with_pressure<0>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 1:   launch_with_pressure<1>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 2:   launch_with_pressure<2>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 4:   launch_with_pressure<4>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 8:   launch_with_pressure<8>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 16:  launch_with_pressure<16>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 32:  launch_with_pressure<32>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 64:  launch_with_pressure<64>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 128: launch_with_pressure<128>(register_pressure, gridDim, blockDim, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        default:
            cerr << "Unsupported arithmetic intensity: " << arith_intensity << ". Please add a case for it." << endl;
            exit(EXIT_FAILURE);
    }
}


int main(int argc, char* argv[]) {
    if (argc != 8) {
        cerr << "Usage: " << argv[0] << " <arith_intensity> <register_pressure> <iterations> <array_size_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }
    // Parse arguments
    int arith_intensity = stoi(argv[1]);
    int register_pressure = stoi(argv[2]); // Parse the new parameter
    int iterations = stoi(argv[3]);
    size_t array_size_mb = stoull(argv[4]);
    size_t shmem_bytes = static_cast<size_t>(stoll(argv[5])); // Shared memory to request per block
    int threads_per_block = stoi(argv[6]);
    int num_blocks = stoi(argv[7]);

    // --- GPU Information ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0)); // Get properties of default device (ID 0)
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
              << ", warpSize: " << prop.warpSize << "\n";

    assert((threads_per_block % prop.warpSize) == 0 && "threads per block must be multiple of warpSize");

    // --- Pointer Chasing Array Setup (Device) ---
    size_t array_bytes = array_size_mb * 1024 * 1024;   // Total array size in bytes
    size_t num_elements = array_bytes / sizeof(uintptr_t); // Number of pointer elements in the array
    cout << "Setting up pointer-chasing array of " << array_size_mb << " MB (" << num_elements << " elements)..." << endl;

    uintptr_t *d_ptr_array = nullptr; // Device pointer for the array
    CUDA_CHECK(cudaMalloc(&d_ptr_array, array_bytes)); // Allocate memory on the GPU

    // --- Pointer Chasing Array Setup (Host) ---
    // Create the pointer chain on the host first.
    vector<uintptr_t> h_ptr_array(num_elements); // Host copy of the array
    vector<size_t> indices(num_elements);       // Helper vector for shuffling indices
    iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ...

    // Shuffle the indices randomly to create a pseudo-random pointer chain.
    mt19937 g(1337); // Mersenne Twister engine seeded for reproducibility
    shuffle(indices.begin(), indices.end(), g);

    // Create the chain: element at shuffled index i points to element at shuffled index i+1.
    // The value stored is the *device* address.
    for (size_t i = 0; i < num_elements - 1; ++i) {
        // h_ptr_array[index pointed from] = device_address_of(index pointed to)
        h_ptr_array[indices[i]] = (uintptr_t)&d_ptr_array[indices[i+1]];
    }
    // Make the last element point back to the first to close the loop (though not strictly necessary for linear chase).
    h_ptr_array[indices[num_elements - 1]] = (uintptr_t)&d_ptr_array[indices[0]];

    // Copy the host array (containing device pointers) to the device array.
    CUDA_CHECK(cudaMemcpy(d_ptr_array, h_ptr_array.data(), array_bytes, cudaMemcpyHostToDevice));

    // --- Starting Addresses Setup ---
    // Prepare the array of unique starting addresses for each thread.
    int total_threads = num_blocks * threads_per_block;
    vector<uintptr_t> h_start_addrs(total_threads); // Host array for start addresses
    // Calculate a stride to spread starting points across the shuffled array, reducing conflicts.
    const size_t stride = num_elements / total_threads;
    for(int i = 0; i < total_threads; ++i) {
        // Assign starting address based on stride through the shuffled indices.
        h_start_addrs[i] = (uintptr_t)&d_ptr_array[indices[(size_t)i * stride]];
    }

    uintptr_t *d_start_addrs = nullptr; // Device pointer for start addresses
    CUDA_CHECK(cudaMalloc(&d_start_addrs, total_threads * sizeof(uintptr_t))); // Allocate on device
    // Copy start addresses from host to device.
    CUDA_CHECK(cudaMemcpy(d_start_addrs, h_start_addrs.data(), total_threads * sizeof(uintptr_t), cudaMemcpyHostToDevice));

    // --- Kernel Launch Setup ---
    dim3 threadsPerBlock(threads_per_block); // CUDA thread block dimensions
    dim3 numBlocksGrid(num_blocks);          // CUDA grid dimensions
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block; // Total number of warps to launch

    WarpEvent *d_warp_events = nullptr; // Device pointer for warp event data
    // Allocate space for event data for every warp.
    CUDA_CHECK(cudaMalloc(&d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));
    // Initialize event data to zero.
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));

    // Create CUDA events for timing the kernel execution on the host side.
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // --- Warmup Launch ---
    // Execute the kernel once before the timed run. This helps ensure caches are warm,
    // GPU clocks might ramp up, and initialization overheads don't affect the measurement.
    cout << "Launching kernel with a=" << arith_intensity << ", pressure=" << register_pressure << ", iterations=" << iterations << "..." << endl;
    // Call the main launcher function, passing the register pressure.
    launch_kernel(arith_intensity, register_pressure, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
    CUDA_CHECK(cudaGetLastError()); // Check for errors after launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for warmup kernel to finish

    // --- Timed Kernel Launch ---
    CUDA_CHECK(cudaEventRecord(ev_start)); // Record start event
    // Launch the kernel again for measurement.
    launch_kernel(arith_intensity, register_pressure, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
    CUDA_CHECK(cudaGetLastError()); // Check for errors after launch
    CUDA_CHECK(cudaEventRecord(ev_stop)); // Record stop event
    CUDA_CHECK(cudaEventSynchronize(ev_stop)); // Wait for kernel and stop event recording to finish

    float milliseconds = 0.0f; // Variable to store elapsed time
    // Calculate the time elapsed between the start and stop events.
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    // --- Process Results ---
    // Copy the WarpEvent data (timestamps, SM IDs) from device back to host.
    vector<WarpEvent> h_events(static_cast<size_t>(totalWarps));
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent), cudaMemcpyDeviceToHost));

    // Define the wrap-around value for the 32-bit clock() counter.
    const uint64_t CLOCK_WRAP = (uint64_t)UINT32_MAX + 1ULL;
    // Helper struct to store timestamp events (start or end) for sorting.
    struct Timestamp {
        uint64_t time; // 64-bit timestamp (accounts for wrap-around)
        int type;      // +1 for start, -1 for end
        bool operator<(Timestamp const& o) const { return time < o.time; } // For sorting
    };

    // Group events by the SM they occurred on.
    map<unsigned int, vector<Timestamp>> events_by_sm;
    for (int w = 0; w < totalWarps; ++w) {
        // Skip warps that didn't record data (might happen with small launches).
        if (h_events[w].start_clock == 0 && h_events[w].end_clock == 0) continue;
        // Convert 32-bit clocks to 64-bit, handling potential wrap-around.
        uint64_t s64 = h_events[w].start_clock;
        uint64_t e64 = h_events[w].end_clock;
        if (e64 < s64) e64 += CLOCK_WRAP; // Add wrap value if end clock is smaller than start
        // Add start and end events to the map for the corresponding SM.
        events_by_sm[h_events[w].sm_id].push_back({s64, +1}); // +1 marks a warp starting
        events_by_sm[h_events[w].sm_id].push_back({e64, -1}); // -1 marks a warp ending
    }

    // --- Calculate Maximum Attained Occupancy ---
    // Iterate through each SM's events to find the peak concurrent warps.
    int overall_max_occ = 0; // Tracks the highest occupancy seen across all SMs
    for (auto &kv : events_by_sm) { // kv is a pair: {sm_id, vector<Timestamp>}
        if (kv.second.empty()) continue; // Skip SMs with no events
        // Sort events chronologically.
        sort(kv.second.begin(), kv.second.end());
        int current_occupancy = 0; // Tracks active warps on this SM
        int max_occupancy_this_sm = 0; // Tracks peak occupancy on this SM
        // Sweep through the sorted events.
        for (auto &ts : kv.second) {
            current_occupancy += ts.type; // Increment for start, decrement for end
            if (current_occupancy > max_occupancy_this_sm) {
                max_occupancy_this_sm = current_occupancy; // Update peak if needed
            }
        }
        // Update the overall maximum if this SM's peak was higher.
        if (max_occupancy_this_sm > overall_max_occ) {
            overall_max_occ = max_occupancy_this_sm;
        }
    }

    // --- Print Summary ---
    // Calculate total operations (1 load + ARITH_INTENSITY adds per iteration).
    // Note: This calculation ignores the extra ops from REGISTER_PRESSURE. If analyzing
    // throughput including those, this needs adjustment. For Volkov's simplistic
    // workload analysis (pressure=0), this is correct.
    double ops_per_warp = static_cast<double>(iterations) * (1.0 + arith_intensity);
    double total_ops = ops_per_warp * totalWarps;
    double time_s = static_cast<double>(milliseconds) / 1000.0; // Convert ms to seconds
    double gops = total_ops / (time_s * 1e9); // Calculate Giga-Ops per second

    // Print results in a parseable format for the script.
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "ArithmeticIntensity: " << arith_intensity << "\n";
    cout << "RegisterPressure: " << register_pressure << "\n"; // Include register pressure in output
    cout << "IterationsPerWarp: " << iterations << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GOps: " << std::fixed << gops << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << overall_max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    // Free all allocated device memory and destroy CUDA events.
    CUDA_CHECK(cudaFree(d_ptr_array));
    CUDA_CHECK(cudaFree(d_start_addrs));
    CUDA_CHECK(cudaFree(d_warp_events));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0; // Indicate successful execution
}