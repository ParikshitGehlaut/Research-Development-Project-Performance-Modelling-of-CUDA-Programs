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
    float f;   // Floating-point representation of the same bits
};

template<int ARITH_INTENSITY> __global__ void simplistic_kernel(const uintptr_t* ptr_array,
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

    volatile float p_as_float = p_union.f; // a
    const float zero = 0.0; // b

    for(int i=0; i<iterations; i++){
        // one memory load operation i.e a = memory[a]
        p_union.f = p_as_float; // reading GPU memory address as float
        uintptr_t next_ptr = *reinterpret_cast<const uintptr_t*>(p_union.ptr); // dereferencing it
        p_union.ptr = next_ptr;

        // Alpha (AI) arithmetic instructions i.e a = a + b
        #pragma unroll // Hint to the compiler to unroll this small loop.
        for (int j = 0; j < ARITH_INTENSITY; ++j) {
            p_as_float = p_as_float + zero;
        }
    }

    if (lane_id == 0) {
        __threadfence_system();
        warp_events[global_warp_id].end_clock = clock(); // Record end time
    }
}


void launch_kernel(int arith_intensity, dim3 gridDim, dim3 blockDim, size_t shmem_bytes,
                   const uintptr_t* d_ptr_array, const uintptr_t* d_start_addrs,
                   int iterations, WarpEvent* d_warp_events)
{
    switch (arith_intensity) {
        case 0:   simplistic_kernel<0><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 1:   simplistic_kernel<1><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 2:   simplistic_kernel<2><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 4:   simplistic_kernel<4><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 8:   simplistic_kernel<8><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 16:  simplistic_kernel<16><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 32:  simplistic_kernel<32><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 64:  simplistic_kernel<64><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        case 128: simplistic_kernel<128><<<gridDim, blockDim, shmem_bytes>>>(d_ptr_array, d_start_addrs, iterations, d_warp_events); break;
        default:
            cerr << "Unsupported arithmetic intensity: " << arith_intensity << endl;
            exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaGetLastError());
}


int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <arith_intensity> <iterations> <array_size_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }

    int arith_intensity = stoi(argv[1]);
    int iterations = stoi(argv[2]);
    size_t array_size_mb = stoull(argv[3]);
    size_t shmem_bytes = static_cast<size_t>(stoll(argv[4]));
    int threads_per_block = stoi(argv[5]);
    int num_blocks = stoi(argv[6]);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
         << ", warpSize: " << prop.warpSize << "\n";

    assert((threads_per_block % prop.warpSize) == 0);

    size_t array_bytes = array_size_mb * 1024 * 1024;
    size_t num_elements = array_bytes / sizeof(uintptr_t);
    cout << "Setting up pointer-chasing array of " << array_size_mb << " MB (" << num_elements << " elements)..." << endl;

    uintptr_t *d_ptr_array = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr_array, array_bytes));

    vector<uintptr_t> h_ptr_array(num_elements);
    vector<size_t> indices(num_elements);
    iota(indices.begin(), indices.end(), 0);

    mt19937 g(1337);
    shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < num_elements - 1; ++i) {
        h_ptr_array[indices[i]] = (uintptr_t)&d_ptr_array[indices[i+1]];
    }
    h_ptr_array[indices[num_elements - 1]] = (uintptr_t)&d_ptr_array[indices[0]];

    CUDA_CHECK(cudaMemcpy(d_ptr_array, h_ptr_array.data(), array_bytes, cudaMemcpyHostToDevice));

    int total_threads = num_blocks * threads_per_block;
    vector<uintptr_t> h_start_addrs(total_threads);
    const size_t stride = num_elements / total_threads;
    for (int i = 0; i < total_threads; ++i) {
        h_start_addrs[i] = (uintptr_t)&d_ptr_array[indices[(size_t)i * stride]];
    }

    uintptr_t *d_start_addrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_start_addrs, total_threads * sizeof(uintptr_t)));
    CUDA_CHECK(cudaMemcpy(d_start_addrs, h_start_addrs.data(), total_threads * sizeof(uintptr_t), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocksGrid(num_blocks);
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block;

    WarpEvent *d_warp_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpEvent)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    cout << "Launching kernel with a=" << arith_intensity << ", iterations=" << iterations << "..." << endl;
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    launch_kernel(arith_intensity, numBlocksGrid, threadsPerBlock, shmem_bytes, d_ptr_array, d_start_addrs, iterations, d_warp_events);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    vector<WarpEvent> h_events(static_cast<size_t>(totalWarps));
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpEvent), cudaMemcpyDeviceToHost));

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
        sort(kv.second.begin(), kv.second.end());
        int current = 0, max_occ = 0;
        for (auto &ts : kv.second) {
            current += ts.type;
            max_occ = max(max_occ, current);
        }
        overall_max_occ = max(overall_max_occ, max_occ);
    }

    double ops_per_warp = static_cast<double>(iterations) * (1.0 + arith_intensity);
    double total_ops = ops_per_warp * totalWarps;
    double time_s = static_cast<double>(milliseconds) / 1000.0;
    double gops = total_ops / (time_s * 1e9);

    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "ArithmeticIntensity: " << arith_intensity << "\n";
    cout << "IterationsPerWarp: " << iterations << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GOps: " << std::fixed << gops << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << overall_max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    CUDA_CHECK(cudaFree(d_ptr_array));
    CUDA_CHECK(cudaFree(d_start_addrs));
    CUDA_CHECK(cudaFree(d_warp_events));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}