/*
    Workload : Vasily Volkov's Latency Measurement (Chapter 6)
    Author   : Parikshit Gehlaut
    Access   : Streaming, Coalesced, Dependent Loads
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
#include <limits>
#include <iomanip>

using namespace std;

#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(_e) << std::endl; \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// device function to get ID of SM
static __device__ __forceinline__ unsigned smid(){
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

// Structure to store the start clock, end clock, and SM ID for a single warp
struct WarpResult {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
    uintptr_t sink; // To prevent compiler optimization
};

/*
    This kernel implements the core latency test as described:
    1. Instruction Mix (Req #1): The loop contains *only* the dependent
       pointer-chasing load.
    2. Access Pattern (Req #4): All threads in the warp are active,
       each chasing its own pointer chain. The host setup ensures
       these accesses are coalesced.
    3. Sink: A warp-wide reduction (sum) is used on the final pointer.
       This *forces* all 32 threads to complete the loop,
       preventing the compiler from optimizing away their work.
*/
__global__ void latency_kernel(const uintptr_t* __restrict__ start_addrs,
                               int iterations,
                               WarpResult* warp_events)
{
    // Shared memory is allocated by the launcher to control occupancy
    extern __shared__ int shmem[]; 
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int block_linear_id = blockIdx.x;
    int warp_global_id = block_linear_id * warps_per_block + warp_id_in_block;

    // All threads load their unique starting pointer
    uintptr_t curr_ptr = start_addrs[global_tid];
    
    // Lane 0 records start time and SM ID
    if(lane_id == 0){
        warp_events[warp_global_id].sm_id = smid();
        // Fence to ensure sm_id is written before clock
        __threadfence_block(); 
        warp_events[warp_global_id].start_clock = clock();
    }

    // (Req #1) The core dependent load loop
    // Pragma unroll 1 is critical to ensure a dependent load chain
    #pragma unroll 1 
    for(int i=0; i<iterations; i++){
        // All threads in the warp issue a dependent load
        // --- FIXED ---
        // Use a C-style cast to correctly apply __restrict__ and fix warning
        curr_ptr = *((const uintptr_t* __restrict__)curr_ptr);
        // --- END FIX ---
    }
    
    // Warp-wide reduction to sink *all* thread results
    unsigned long long ptr_sum = (unsigned long long)curr_ptr;
    #pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        ptr_sum += __shfl_down_sync(0xFFFFFFFF, ptr_sum, offset);
    }

    // Lane 0 records end time and writes the final summed sink
    if (lane_id == 0) {
        warp_events[warp_global_id].end_clock = clock();
        warp_events[warp_global_id].sink = (uintptr_t)ptr_sum;
        shmem[0] = (int)ptr_sum;
    }
}


// Host Code
int main(int argc, char* argv[]) {
    // --- Argument Parsing ---
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <iterations> <array_size_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }
    int iterations = std::stoi(argv[1]); 
    size_t array_size_mb = std::stoull(argv[2]);
    size_t shmem_bytes = static_cast<size_t>(std::stoll(argv[3]));
    int threads_per_block = std::stoi(argv[4]);
    int num_blocks = std::stoi(argv[5]);

    // --- GPU Information ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
              << ", warpSize: " << prop.warpSize << ", Clock: " << (prop.clockRate / 1e6) << " GHz\n";
    assert((threads_per_block % prop.warpSize) == 0 && "threads per block must be multiple of warpSize");

    // --- Pointer Chasing Array Setup (Device) ---
    size_t array_bytes = array_size_mb * 1024 * 1024;
    size_t num_elements = array_bytes / sizeof(uintptr_t);
    cout << "Setting up pointer-chasing array of " << array_size_mb << " MB (" << num_elements << " elements)..." << endl;

    // (Req #4) N = thread block size
    const size_t N = static_cast<size_t>(threads_per_block);

    // The highest starting thread index is (num_blocks * threads_per_block - 1)
    // But we partition blocks by N, so the highest start index for a thread is:
    // Start index = i + j*N 
    // Max j = num_blocks - 1
    // Max i = threads_per_block - 1 (which is N - 1)
    size_t max_start_index = (static_cast<size_t>(num_blocks - 1) * N) + (N - 1);
    
    // This thread's chain goes for 'iterations' steps, with each step
    // adding N. The last element it *accesses* is:
    // max_start_index + (iterations - 1) * N
    size_t max_idx_accessed_by_kernel = max_start_index + (static_cast<size_t>(iterations - 1) * N);
    
    // The element at 'max_idx_accessed_by_kernel' must point to
    // (max_idx_accessed_by_kernel + N), and *that* must be in the array.
    if ((max_idx_accessed_by_kernel + N) >= num_elements) {
        cerr << "ERROR: Kernel execution will access memory out of bounds." << endl;
        cerr << "Total elements needed for non-looping chain: " << (max_idx_accessed_by_kernel + N + 1) << endl;
        cerr << "Array size: " << num_elements << endl;
        cerr << "Reduce iterations, blocks, or increase array size." << endl;
        return 1;
    }

    uintptr_t *d_ptr_array = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr_array, array_bytes));

    // --- Pointer Chasing Array Setup (Host) ---
    vector<uintptr_t> h_ptr_array(num_elements);
    cout << "Setting up NON-LOOPING (linear) pointer chain with stride N = " << N << "..." << endl;

    // (Req #4) Create the chain: element i points to element i+N
    // This logic is correct and remains unchanged.
    for (size_t i = 0; i < num_elements; i++) {
        size_t target_idx = i + N;
        if (target_idx >= num_elements) {
            h_ptr_array[i] = 0; // End of chain
        } else {
            h_ptr_array[i] = reinterpret_cast<uintptr_t>(d_ptr_array + target_idx);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_ptr_array, h_ptr_array.data(), array_bytes, cudaMemcpyHostToDevice));

    // --- Starting Addresses Setup ---
    int total_threads = num_blocks * threads_per_block;
    vector<uintptr_t> h_start_addrs(total_threads);

    // We space blocks by N (threads_per_block), not by (iterations * N)
    cout << "Setting start addresses with block spacing = " << N << " elements." << endl;

    for (int j = 0; j < num_blocks; ++j) { // blockIdx.x
        for (int i = 0; i < threads_per_block; ++i) { // threadIdx.x
            int global_tid = j * threads_per_block + i;
            
            // Start index = i + j*N
            // Each block starts *after* the previous one's slice.
            size_t start_index = (static_cast<size_t>(j) * N) + static_cast<size_t>(i);
            
            h_start_addrs[global_tid] = reinterpret_cast<uintptr_t>(d_ptr_array + start_index);
        }
    }

    uintptr_t *d_start_addrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_start_addrs, total_threads * sizeof(uintptr_t)));
    CUDA_CHECK(cudaMemcpy(d_start_addrs, h_start_addrs.data(), total_threads * sizeof(uintptr_t), cudaMemcpyHostToDevice));

    // --- Kernel Launch Setup ---
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocksGrid(num_blocks);
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block;

    WarpResult *d_warp_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));

    // --- Set Shared Memory Attribute ---
    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shmem_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));

    // --- Warmup Launch ---
    cout << "Launching kernel with iterations=" << iterations 
              << ", shmem=" << shmem_bytes << "..." << endl;
    latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(
        d_start_addrs, 1, d_warp_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed Kernel Launch ---
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(
        d_start_addrs, iterations, d_warp_events);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Process Results (Req #6) ---
    vector<WarpResult> h_events(static_cast<size_t>(totalWarps));
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpResult), cudaMemcpyDeviceToHost));

    map<unsigned int, vector<WarpResult>> sm_results;
    for (const auto& res : h_events) {
        if (res.sm_id < (unsigned int)prop.multiProcessorCount && res.end_clock > res.start_clock) {
             sm_results[res.sm_id].push_back(res);
        }
    }
    
    // --- Process Results (Modified for Global Mean) ---
    uint64_t global_total_cycles = 0;
    size_t global_total_warps = 0;
    const uint64_t CLOCK_WRAP = (uint64_t)UINT32_MAX + 1ULL;
    
    // We still track min_latency just for observation, though not used for the model base
    double global_min_latency_per_op = std::numeric_limits<double>::max();

    for (auto const& [smid, results_vec] : sm_results) {
        if (results_vec.empty()) continue;

        for (const auto& res : results_vec) {
            uint64_t duration = (uint64_t)res.end_clock - (uint64_t)res.start_clock;
            
            // Handle clock wraparound
            if (res.end_clock < res.start_clock) {
                duration += CLOCK_WRAP;
            }
            
            // Add to global accumulators
            global_total_cycles += duration;
            global_total_warps++;
            
            // Track absolute minimum (single warp) just for reference
            double current_latency_op = (double)duration / iterations;
            if (current_latency_op < global_min_latency_per_op) {
                global_min_latency_per_op = current_latency_op;
            }
        }
    }

    double global_mean_latency = 0;
    if (global_total_warps > 0) {
        // 1. Average cycles per warp
        double avg_cycles_per_warp = (double)global_total_cycles / global_total_warps;
        // 2. Normalize by instructions (iterations)
        global_mean_latency = avg_cycles_per_warp / iterations;
    } else {
        global_min_latency_per_op = 0;
    }

    // --- Print Summary ---
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "Iterations: " << iterations << "\n";
    cout << "ArraySize_MB: " << array_size_mb << "\n";
    cout << "Shmem_Bytes: " << shmem_bytes << "\n";
    cout << "ThreadsPerBlock: " << threads_per_block << "\n";
    cout << "NumBlocks: " << num_blocks << "\n";
    cout << "TotalWarpsMeasured: " << global_total_warps << "\n";
    // This is the value to use for Volkov's model
    cout << "MeanLatency_cycles: " << std::fixed << std::setprecision(2) << global_mean_latency << "\n";
    // This is just for info (Section 6.6 of Volkov)
    cout << "MinLatency_cycles: " << std::fixed << std::setprecision(2) << global_min_latency_per_op << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_ptr_array));
    CUDA_CHECK(cudaFree(d_start_addrs));
    CUDA_CHECK(cudaFree(d_warp_events));

    return 0;
}