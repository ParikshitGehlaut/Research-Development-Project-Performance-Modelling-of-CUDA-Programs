/*
    Workload : Memory Latency Measurement (Volkov's Methodology)
    Author   : Parikshit Gehlaut
    Access   : Streaming, Coalesced, Dependent 4-byte Loads (uint32_t offsets)

    This kernel measures unloaded global memory latency using pointer chasing
    with 4-byte (uint32_t) element indices, replicating Volkov's 32-bit load
    approach on modern 64-bit GPUs.

    Each warp load = 32 threads × 4 bytes = 128 bytes (identical to Volkov).
*/

#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <map>
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

// Device function to get SM ID
static __device__ __forceinline__ unsigned smid(){
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

// Per-warp measurement result
struct WarpResult {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
    uint32_t sink;  // Prevent compiler optimization
};

/*
    Memory Latency Kernel (4-byte offset chase)

    Each thread chases a chain of uint32_t indices:
      curr_idx = d_indices[curr_idx]
    This creates a dependent load chain with 4 bytes per thread per load.

    Array layout: d_indices[i] = i + N (linear stride, non-looping)
    where N = threads_per_block, giving coalesced warp-level accesses.
*/
__global__ void mem_lat_kernel(const uint32_t* __restrict__ d_indices,
                               const uint32_t* __restrict__ start_indices,
                               int iterations,
                               WarpResult* warp_events)
{
    extern __shared__ int shmem[];

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Each thread loads its starting index (4 bytes)
    uint32_t curr_idx = start_indices[global_tid];

    // Lane 0 records start time and SM ID
    if(lane_id == 0){
        warp_events[warp_global_id].sm_id = smid();
        __threadfence_block();
        warp_events[warp_global_id].start_clock = clock();
    }

    // Core dependent load loop: each load depends on the previous result
    #pragma unroll 1
    for(int i = 0; i < iterations; i++){
        // 4-byte dependent load — Volkov's methodology
        curr_idx = d_indices[curr_idx];
    }

    // Warp-wide reduction to sink ALL thread results
    unsigned long long idx_sum = (unsigned long long)curr_idx;
    #pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        idx_sum += __shfl_down_sync(0xFFFFFFFF, idx_sum, offset);
    }

    // Lane 0 records end time and writes sink
    if (lane_id == 0) {
        warp_events[warp_global_id].end_clock = clock();
        warp_events[warp_global_id].sink = (uint32_t)idx_sum;
        shmem[0] = (int)idx_sum;
    }
}


// Host Code
int main(int argc, char* argv[]) {
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

    // --- Index Array Setup ---
    // Array of uint32_t indices (4 bytes each, matching Volkov's 32-bit pointer loads)
    size_t array_bytes = array_size_mb * 1024 * 1024;
    size_t num_elements = array_bytes / sizeof(uint32_t);  // 4 bytes per element
    cout << "Setting up index-chasing array of " << array_size_mb << " MB (" << num_elements << " uint32_t elements)..." << endl;

    const size_t N = static_cast<size_t>(threads_per_block);

    // Validate bounds: last access = start_index + (iterations-1)*N
    size_t max_start_index = (static_cast<size_t>(num_blocks - 1) * N) + (N - 1);
    size_t max_idx_accessed = max_start_index + (static_cast<size_t>(iterations - 1) * N);

    if ((max_idx_accessed + N) >= num_elements) {
        cerr << "ERROR: Kernel execution will access out of bounds." << endl;
        cerr << "Elements needed: " << (max_idx_accessed + N + 1) << ", Array size: " << num_elements << endl;
        cerr << "Reduce iterations, blocks, or increase array size." << endl;
        return 1;
    }

    // Allocate device array
    uint32_t *d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, num_elements * sizeof(uint32_t)));

    // Build index chain on host: element[i] = i + N (linear stride, non-looping)
    vector<uint32_t> h_indices(num_elements);
    cout << "Setting up NON-LOOPING (linear) index chain with stride N = " << N << "..." << endl;

    for (size_t i = 0; i < num_elements; i++) {
        size_t target_idx = i + N;
        if (target_idx >= num_elements) {
            h_indices[i] = 0;  // End of chain (never reached during measured run)
        } else {
            h_indices[i] = static_cast<uint32_t>(target_idx);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- Starting Indices ---
    int total_threads = num_blocks * threads_per_block;
    vector<uint32_t> h_start_indices(total_threads);

    for (int j = 0; j < num_blocks; ++j) {
        for (int i = 0; i < threads_per_block; ++i) {
            int global_tid = j * threads_per_block + i;
            size_t start_index = (static_cast<size_t>(j) * N) + static_cast<size_t>(i);
            h_start_indices[global_tid] = static_cast<uint32_t>(start_index);
        }
    }

    uint32_t *d_start_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_start_indices, total_threads * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_start_indices, h_start_indices.data(), total_threads * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- Kernel Launch Setup ---
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocksGrid(num_blocks);
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block;

    WarpResult *d_warp_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));

    // Set shared memory attribute
    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(mem_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));

    // --- Warmup ---
    cout << "Launching kernel with iterations=" << iterations << ", shmem=" << shmem_bytes << "..." << endl;
    mem_lat_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(d_indices, d_start_indices, 1, d_warp_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed Launch ---
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaDeviceSynchronize());

    mem_lat_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Process Results ---
    vector<WarpResult> h_events(static_cast<size_t>(totalWarps));
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpResult), cudaMemcpyDeviceToHost));

    map<unsigned int, vector<WarpResult>> sm_results;
    for (const auto& res : h_events) {
        if (res.sm_id < (unsigned int)prop.multiProcessorCount && res.end_clock > res.start_clock) {
             sm_results[res.sm_id].push_back(res);
        }
    }

    uint64_t global_total_cycles = 0;
    size_t global_total_warps = 0;
    const uint64_t CLOCK_WRAP = (uint64_t)UINT32_MAX + 1ULL;
    double global_min_latency_per_op = std::numeric_limits<double>::max();

    for (auto const& [smid, results_vec] : sm_results) {
        if (results_vec.empty()) continue;
        for (const auto& res : results_vec) {
            uint64_t duration = (uint64_t)res.end_clock - (uint64_t)res.start_clock;
            if (res.end_clock < res.start_clock) {
                duration += CLOCK_WRAP;
            }
            global_total_cycles += duration;
            global_total_warps++;
            double current_latency_op = (double)duration / iterations;
            if (current_latency_op < global_min_latency_per_op) {
                global_min_latency_per_op = current_latency_op;
            }
        }
    }

    double global_mean_latency = 0;
    if (global_total_warps > 0) {
        double avg_cycles_per_warp = (double)global_total_cycles / global_total_warps;
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
    cout << "MeanLatency_cycles: " << std::fixed << std::setprecision(2) << global_mean_latency << "\n";
    cout << "MinLatency_cycles: " << std::fixed << std::setprecision(2) << global_min_latency_per_op << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_start_indices));
    CUDA_CHECK(cudaFree(d_warp_events));

    return 0;
}
