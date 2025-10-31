/*
    file : Simplistic_kernel.cu
    Author : Parikshit Gehlaut
*/

#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <map>
#include <random>
#include <numeric>
#include <cassert>

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

template<int ARITH_INTENSITY>
__global__ void simplisticKernel(const uintptr_t* ptr_array,
                                      const uintptr_t* start_addrs,
                                      int iterations,
                                      WarpEvent* warp_events)
{
    // Declare shared memory for Occupancy Control
    extern __shared__ int shmem[];
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int block_linear_id = blockIdx.x;
    int warp_global_id = block_linear_id * warps_per_block + warp_id_in_block;


    if(lane_id == 0){
        warp_events[warp_global_id].start_clock = clock();
        warp_events[warp_global_id].sm_id = smid();
    }

    // --------------------  Simplistic Kernel Logic -------------------------

    uintptr_t curr_ptr = start_addrs[global_tid];
    
    const float one_f = 1.0f; 

    uint32_t tmp_int32;
    volatile float tmp_float; 

    for(int i=0; i<iterations; i++){
        
        // 1. LOAD (Dependency on previous iteration's ALU chain)
        curr_ptr = *reinterpret_cast<const uintptr_t*>(curr_ptr);

        // 2. ALU CHAIN (Dependent on the load)
        
        // Extract bits
        tmp_int32 = (uint32_t)(curr_ptr & 0xFFFFFFFFULL);
        // Cast to float. The 'volatile' ensures this write happens.
        tmp_float = __int_as_float(tmp_int32); 

        // Now, FP32 arithmetic chain.
        // This implements the "FMUL R1, R1, R2"
        #pragma unroll
        for (int j = 0; j < ARITH_INTENSITY; j++) {
            tmp_float = tmp_float * one_f;
        }

        tmp_int32 = __float_as_int(tmp_float); 
        
        // The pointer is reconstructed identically, fixing the crash.
        curr_ptr = (curr_ptr & 0xFFFFFFFF00000000ULL) | (uintptr_t)tmp_int32;
    }


    if (lane_id == 0) {
        // prevent compiler optimisation
        shmem[0] = (int)(curr_ptr & 0xFFFFFFFFULL);
        __threadfence_system();
        warp_events[warp_global_id].end_clock = clock();
    }
}


void launch_kernel(int arith_intensity, dim3 gridDim, dim3 blockDim, size_t shmem_bytes,
                   const uintptr_t* d_ptr_array, const uintptr_t* d_start_addrs,
                   int iterations, WarpEvent* d_warp_events)
{
    // Select the correct template instantiation based on the arith_intensity argument.
    switch (arith_intensity) {
        case 0:   simplisticKernel<0><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 1:   simplisticKernel<1><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 2:   simplisticKernel<2><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 4:   simplisticKernel<4><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 8:   simplisticKernel<8><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 16:  simplisticKernel<16><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 32:  simplisticKernel<32><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 64:  simplisticKernel<64><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 128: simplisticKernel<128><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 256: simplisticKernel<256><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 512: simplisticKernel<512><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        default:
            cerr << "Unsupported arithmetic intensity: " << arith_intensity << ". Please add a case for it." << endl;
            exit(EXIT_FAILURE);
    }
}

// Host Code
int main(int argc, char* argv[]) {
    // --- Argument Parsing ---
    // Expect 7 command-line arguments now.
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <arith_intensity> <iterations> <array_size_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }
    // Parse arguments
    int arith_intensity = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]); 
    size_t array_size_mb = std::stoull(argv[3]);
    size_t shmem_bytes = static_cast<size_t>(std::stoll(argv[4]));
    int threads_per_block = std::stoi(argv[5]);
    int num_blocks = std::stoi(argv[6]);

    // --- GPU Information ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0)); // Get properties of default device (ID 0)
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
              << ", warpSize: " << prop.warpSize << "\n";
    // Ensure block dimension is a multiple of warp size for simplicity.
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
    std::mt19937 g(1337); // Mersenne Twister engine seeded for reproducibility
    std::shuffle(indices.begin(), indices.end(), g);

    // Create the chain: element at shuffled index i points to element at shuffled index i+1.
    // The value stored is the *device* address.
    for (size_t i = 0; i < num_elements - 1; i++) {
        // h_ptr_array[index pointed from] = device_address_of(index pointed to)
        h_ptr_array[indices[i]] = reinterpret_cast<uintptr_t>(d_ptr_array + indices[i+1]);
    }
    // Make the last element point back to the first to close the loop (though not strictly necessary for linear chase).
    h_ptr_array[indices[num_elements - 1]] = reinterpret_cast<uintptr_t>(d_ptr_array + indices[0]);

    // Copy the host array (containing device pointers) to the device array.
    CUDA_CHECK(cudaMemcpy(d_ptr_array, h_ptr_array.data(), array_bytes, cudaMemcpyHostToDevice));

    // --- Starting Addresses Setup ---
    // Prepare the array of unique starting addresses for each thread.
    int total_threads = num_blocks * threads_per_block;
    
    // Calculate a stride to spread starting points across the shuffled array, reducing conflicts.
    size_t stride = std::max<size_t>(1, num_elements / static_cast<size_t>(total_threads));
    vector<uintptr_t> h_start_addrs(total_threads);
    for (int i = 0; i < total_threads; ++i) {
        size_t idx = (static_cast<size_t>(i) * stride) % num_elements;
        h_start_addrs[i] = reinterpret_cast<uintptr_t>(d_ptr_array + indices[idx]);
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

    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shmem_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    
    // --- Set Shared Memory Attribute ---
    switch (arith_intensity) {
        case 0:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<0>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 1:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<1>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 2:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<2>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 4:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<4>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 8:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<8>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 16:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<16>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 32:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<32>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 64:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<64>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 128: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<128>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 256: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<256>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 512: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(simplisticKernel<512>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        default:
            cerr << "Unsupported arithmetic intensity for cudaFuncSetAttribute: " << arith_intensity << endl;
            exit(EXIT_FAILURE);
    }

    // --- Warmup Launch ---
    // Execute the kernel once before the timed run. This helps ensure caches are warm,
    // GPU clocks might ramp up, and initialization overheads don't affect the measurement.
    cout << "Launching kernel with a=" << arith_intensity << ", iterations=" << iterations << "..." << endl;
    // Call the main launcher function (no register pressure).
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
    CUDA_CHECK(cudaGetLastError()); // Check for errors after launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for warmup kernel to finish

    // --- Timed Kernel Launch ---
    CUDA_CHECK(cudaEventRecord(ev_start)); // Record start event
    // Launch the kernel again for measurement.
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
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
        std::sort(kv.second.begin(), kv.second.end());
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
    double flops_per_thread = static_cast<double>(iterations) * static_cast<double>(arith_intensity);
    double total_threads_launched = static_cast<double>(num_blocks) * static_cast<double>(threads_per_block);
    double total_flops = flops_per_thread * total_threads_launched;

    double time_s = static_cast<double>(milliseconds) / 1000.0; // Convert ms to seconds
    double gflops = (time_s > 1e-9) ? (total_flops / (time_s * 1e9)) : 0.0; // Avoid division by zero

    // Print results in a parseable format for the script.
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "ArithmeticIntensity: " << arith_intensity << "\n";
    cout << "IterationsPerWarp: " << iterations << "\n"; // Note: This is iterations per thread
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    // Output GFLOPS
    cout << "Throughput_GFLOPS: " << std::fixed << gflops << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << overall_max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_ptr_array));
    CUDA_CHECK(cudaFree(d_start_addrs));
    CUDA_CHECK(cudaFree(d_warp_events));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}