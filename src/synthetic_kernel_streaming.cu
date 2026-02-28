/*
    Workload : Volkov's Synthetic Kernel (Streaming Memory Access)
    Author   : Parikshit Gehlaut
    Access   : Streaming, Coalesced, 4-byte Loads (uint32_t offsets)
    File     : synthetic_kernel_streaming.cu

    Replicates Volkov's synthetic workload for validating the analytical model.
    Uses 4-byte (uint32_t) index-based memory access with configurable
    arithmetic intensity (AI). Each iteration:
      1. Load uint32_t index (4-byte dependent load)
      2. Reinterpret as float
      3. AI × FMUL operations (dependent chain)
      4. Reinterpret back to uint32_t index

    Each warp load = 32 threads × 4 bytes = 128 bytes (matches Volkov exactly).
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

struct WarpEvent {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
};

/*
    Synthetic Kernel with configurable Arithmetic Intensity (AI)

    Loop body per iteration:
      1. curr_idx = d_indices[curr_idx]     → 4-byte load (128 bytes/warp)
      2. tmp_float = __int_as_float(curr_idx)
      3. for j in [0, AI): tmp_float *= 1.0f   → FMUL chain
      4. curr_idx = __float_as_int(tmp_float)

    This is cleaner than the 8-byte version: no bit extraction/merging needed
    because the loaded value (uint32_t) is directly reinterpretable as float.
*/
template<int ARITH_INTENSITY>
__global__ void syntheticKernel(const uint32_t* __restrict__ d_indices,
                                const uint32_t* __restrict__ start_indices,
                                int iterations,
                                WarpEvent* warp_events)
{
    extern __shared__ int shmem[];

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int warp_global_id = blockIdx.x * warps_per_block + warp_id_in_block;

    // Load starting index
    uint32_t curr_idx = start_indices[global_tid];

    const float one_f = 1.0f;
    volatile float tmp_float = 0.0f;

    if(lane_id == 0){
        warp_events[warp_global_id].start_clock = clock();
        warp_events[warp_global_id].sm_id = smid();
    }

    for(int i = 0; i < iterations; i++){

        // 1. LOAD: 4-byte dependent load (matches Volkov's 32-bit pointer chase)
        curr_idx = d_indices[curr_idx];

        // 2. CONVERT: Reinterpret uint32_t as float (no bit extraction needed)
        tmp_float = __int_as_float(curr_idx);

        // 3. COMPUTE: AI × FMUL operations (dependent chain)
        #pragma unroll
        for (int j = 0; j < ARITH_INTENSITY; j++) {
            tmp_float = tmp_float * one_f;
        }

        // 4. CONVERT BACK: Reinterpret float as uint32_t index
        curr_idx = __float_as_int(tmp_float);
    }

    if (lane_id == 0) {
        __threadfence_system();
        warp_events[warp_global_id].end_clock = clock();
        shmem[0] = (int)curr_idx;  // Sink
    }
}


void launch_kernel(int arith_intensity, dim3 gridDim, dim3 blockDim, size_t shmem_bytes,
                   const uint32_t* d_indices, const uint32_t* d_start_indices,
                   int iterations, WarpEvent* d_warp_events)
{
    switch (arith_intensity) {
        case 0:   syntheticKernel<0><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 1:   syntheticKernel<1><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 2:   syntheticKernel<2><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 4:   syntheticKernel<4><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 8:   syntheticKernel<8><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 16:  syntheticKernel<16><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 32:  syntheticKernel<32><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 64:  syntheticKernel<64><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 128: syntheticKernel<128><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 256: syntheticKernel<256><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        case 512: syntheticKernel<512><<<gridDim, blockDim, shmem_bytes>>>(d_indices, d_start_indices, iterations, d_warp_events); break;
        default:
            cerr << "Unsupported arithmetic intensity: " << arith_intensity << ". Please add a case for it." << endl;
            exit(EXIT_FAILURE);
    }
}


// Host Code
int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <arith_intensity> <iterations> <array_size_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }
    int arith_intensity = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]);
    size_t array_size_mb = std::stoull(argv[3]);
    size_t shmem_bytes = static_cast<size_t>(std::stoll(argv[4]));
    int threads_per_block = std::stoi(argv[5]);
    int num_blocks = std::stoi(argv[6]);

    // --- GPU Information ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
              << ", warpSize: " << prop.warpSize << "\n";
    assert((threads_per_block % prop.warpSize) == 0 && "threads per block must be multiple of warpSize");

    // --- Index Array Setup (uint32_t, 4 bytes each) ---
    size_t array_bytes = array_size_mb * 1024 * 1024;
    size_t num_elements = array_bytes / sizeof(uint32_t);
    cout << "Setting up index array of " << array_size_mb << " MB (" << num_elements << " uint32_t elements)..." << endl;

    const size_t N = static_cast<size_t>(threads_per_block);
    const size_t spacing_per_block = static_cast<size_t>(iterations) * N;

    // Validate bounds
    size_t last_in_chain = (static_cast<size_t>(iterations - 1) * N);
    size_t max_idx = (static_cast<size_t>(num_blocks - 1) * spacing_per_block) + last_in_chain + (N - 1);

    cout << "Configuration requires " << (max_idx + 1) << " elements." << endl;
    cout << "Array has " << num_elements << " elements." << endl;

    if ((max_idx + N) >= num_elements) {
        cerr << "ERROR: Out of bounds. Need " << (max_idx + N + 1) << ", have " << num_elements << endl;
        return 1;
    }

    // Allocate and build index chain
    uint32_t *d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, num_elements * sizeof(uint32_t)));

    vector<uint32_t> h_indices(num_elements);
    cout << "Setting up NON-LOOPING (linear) index chain with stride N = " << N << "..." << endl;

    for (size_t i = 0; i < num_elements; i++) {
        size_t target = i + N;
        h_indices[i] = (target >= num_elements) ? 0 : static_cast<uint32_t>(target);
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- Starting Indices ---
    int total_threads = num_blocks * threads_per_block;
    vector<uint32_t> h_start(total_threads);

    cout << "Setting start indices with block spacing = " << spacing_per_block << " elements." << endl;

    for (int j = 0; j < num_blocks; ++j) {
        for (int i = 0; i < threads_per_block; ++i) {
            int tid = j * threads_per_block + i;
            h_start[tid] = static_cast<uint32_t>((static_cast<size_t>(j) * spacing_per_block) + static_cast<size_t>(i));
        }
    }

    uint32_t *d_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_start, total_threads * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_start, h_start.data(), total_threads * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- Kernel Launch Setup ---
    dim3 threadsPerBlock_dim(threads_per_block);
    dim3 numBlocksGrid(num_blocks);
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block;

    WarpEvent *d_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));
    CUDA_CHECK(cudaMemset(d_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Set shared memory attribute for all template instantiations
    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    switch (arith_intensity) {
        case 0:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<0>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 1:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<1>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 2:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<2>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 4:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<4>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 8:   CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<8>),   cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 16:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<16>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 32:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<32>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 64:  CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<64>),  cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 128: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<128>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 256: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<256>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        case 512: CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(syntheticKernel<512>), cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes)); break;
        default:
            cerr << "Unsupported AI for cudaFuncSetAttribute: " << arith_intensity << endl;
            exit(EXIT_FAILURE);
    }

    // --- Warmup ---
    cout << "Launching synthetic kernel with AI=" << arith_intensity << ", iterations=" << iterations << "..." << endl;
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock_dim, shmem_bytes, d_indices, d_start, iterations, d_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed Launch ---
    CUDA_CHECK(cudaEventRecord(ev_start));
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock_dim, shmem_bytes, d_indices, d_start, iterations, d_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    // --- Process Results ---
    vector<WarpEvent> h_events(static_cast<size_t>(totalWarps));
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent), cudaMemcpyDeviceToHost));

    const uint64_t CLOCK_WRAP = (uint64_t)UINT32_MAX + 1ULL;
    struct Timestamp {
        uint64_t time;
        int type;
        bool operator<(Timestamp const& o) const { return time < o.time; }
    };

    map<unsigned int, vector<Timestamp>> events_by_sm;
    for (int w = 0; w < totalWarps; ++w) {
        if (h_events[w].start_clock == 0 && h_events[w].end_clock == 0) continue;
        uint64_t s64 = h_events[w].start_clock;
        uint64_t e64 = h_events[w].end_clock;
        if (e64 < s64) e64 += CLOCK_WRAP;
        events_by_sm[h_events[w].sm_id].push_back({s64, +1});
        events_by_sm[h_events[w].sm_id].push_back({e64, -1});
    }

    int overall_max_occ = 0;
    for (auto &kv : events_by_sm) {
        if (kv.second.empty()) continue;
        std::sort(kv.second.begin(), kv.second.end());
        int curr = 0, max_occ = 0;
        for (auto &ts : kv.second) {
            curr += ts.type;
            if (curr > max_occ) max_occ = curr;
        }
        if (max_occ > overall_max_occ) overall_max_occ = max_occ;
    }

    // --- Compute GFLOPS and throughput ---
    double flops_per_thread = static_cast<double>(iterations) * static_cast<double>(arith_intensity);
    double total_flops = flops_per_thread * static_cast<double>(total_threads);
    double time_s = static_cast<double>(milliseconds) / 1000.0;
    double gflops = (time_s > 1e-9) ? (total_flops / (time_s * 1e9)) : 0.0;

    // Memory throughput: 4 bytes per thread per iteration (uint32_t)
    double bytes_per_access = sizeof(uint32_t);  // 4 bytes — Volkov's load width
    double total_bytes = static_cast<double>(total_threads) * static_cast<double>(iterations) * bytes_per_access;
    double throughput_GBps = (time_s > 1e-9) ? (total_bytes / time_s / (1024.0 * 1024.0 * 1024.0)) : 0.0;

    // --- Print Summary ---
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "ArithmeticIntensity: " << arith_intensity << "\n";
    cout << "IterationsPerWarp: " << iterations << "\n";
    cout << "ArraySize_MB: " << array_size_mb << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GFLOPS: " << std::fixed << gflops << "\n";
    cout << "Throughput_GBps: " << std::fixed << throughput_GBps << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << overall_max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_events));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}
