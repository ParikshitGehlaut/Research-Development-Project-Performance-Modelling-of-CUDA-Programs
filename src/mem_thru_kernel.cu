/*
    Workload : Effective Memory Throughput Measurement (Volkov's Methodology)
    Author   : Parikshit Gehlaut
    Access   : Streaming, Coalesced, 4-byte Loads (uint32_t)
    File     : mem_thru_kernel.cu

    Measures peak effective memory bandwidth by saturating the memory bus
    with coalesced 4-byte (uint32_t) streaming loads. Each warp load = 128 bytes.
    
    This kernel uses the synthetic kernel structure at AI=0: pure memory loads
    with no arithmetic, to isolate the memory throughput bound.
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

struct WarpEvent {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
};

/*
    Memory Throughput Kernel (4-byte streaming loads)

    Each thread chases a linear index chain:
      curr_idx = d_indices[curr_idx]
    where d_indices[i] = i + N (stride = threads_per_block).

    With AI=0 (no arithmetic), this isolates the memory throughput bound.
    The chain gives coalesced, non-repeating, streaming access.
*/
__global__ void mem_thru_kernel(const uint32_t* __restrict__ d_indices,
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

    // Load starting index (4 bytes)
    uint32_t curr_idx = start_indices[global_tid];

    if(lane_id == 0){
        warp_events[warp_global_id].start_clock = clock();
        warp_events[warp_global_id].sm_id = smid();
    }

    // Pure memory chase — no arithmetic (AI=0)
    #pragma unroll 1
    for(int i = 0; i < iterations; i++){
        curr_idx = d_indices[curr_idx];  // 4-byte dependent load
    }

    if (lane_id == 0) {
        __threadfence_system();
        warp_events[warp_global_id].end_clock = clock();
        shmem[0] = (int)curr_idx;  // Sink to prevent optimization
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
              << ", warpSize: " << prop.warpSize << "\n";
    assert((threads_per_block % prop.warpSize) == 0 && "threads per block must be multiple of warpSize");

    // --- Index Array Setup (uint32_t, 4 bytes each) ---
    size_t array_bytes = array_size_mb * 1024 * 1024;
    size_t num_elements = array_bytes / sizeof(uint32_t);
    cout << "Setting up streaming index array of " << array_size_mb << " MB (" << num_elements << " uint32_t elements)..." << endl;

    const size_t N = static_cast<size_t>(threads_per_block);
    // Spacing: each block sweeps iterations * N elements
    const size_t spacing_per_block = static_cast<size_t>(iterations) * N;

    // Validate bounds
    size_t last_element_in_chain = (static_cast<size_t>(iterations - 1) * N);
    size_t max_idx_accessed = (static_cast<size_t>(num_blocks - 1) * spacing_per_block) + last_element_in_chain + (N - 1);

    if ((max_idx_accessed + N) >= num_elements) {
        cerr << "ERROR: Out of bounds. Need " << (max_idx_accessed + N + 1) << " elements, have " << num_elements << endl;
        cerr << "Reduce iterations/blocks or increase array size." << endl;
        return 1;
    }

    // Allocate and build index chain
    uint32_t *d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, num_elements * sizeof(uint32_t)));

    vector<uint32_t> h_indices(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        size_t target = i + N;
        h_indices[i] = (target >= num_elements) ? 0 : static_cast<uint32_t>(target);
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // --- Starting Indices ---
    int total_threads = num_blocks * threads_per_block;
    vector<uint32_t> h_start(total_threads);

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

    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(mem_thru_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));

    // --- Warmup ---
    cout << "Launching mem_thru kernel with iterations=" << iterations << "..." << endl;
    mem_thru_kernel<<<numBlocksGrid, threadsPerBlock_dim, shmem_bytes>>>(d_indices, d_start, iterations, d_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed Launch ---
    CUDA_CHECK(cudaEventRecord(ev_start));
    mem_thru_kernel<<<numBlocksGrid, threadsPerBlock_dim, shmem_bytes>>>(d_indices, d_start, iterations, d_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    // --- Calculate throughput ---
    // Each thread does 'iterations' loads of 4 bytes (uint32_t)
    double bytes_per_thread_access = sizeof(uint32_t);  // 4 bytes — Volkov's load width
    double total_bytes_accessed = static_cast<double>(total_threads) * static_cast<double>(iterations) * bytes_per_thread_access;
    double time_s = static_cast<double>(milliseconds) / 1000.0;
    double throughput_GBps = 0.0;
    if (time_s > 1e-9) {
        throughput_GBps = total_bytes_accessed / time_s / (1024.0 * 1024.0 * 1024.0);
    }

    // --- Process occupancy from warp events ---
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

    // --- Print Summary ---
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "Iterations: " << iterations << "\n";
    cout << "ArraySize_MB: " << array_size_mb << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GBps: " << std::fixed << std::setprecision(4) << throughput_GBps << "\n";
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
