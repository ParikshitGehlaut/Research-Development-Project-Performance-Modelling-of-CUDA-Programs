/*
    Workload: Naive Matrix Multiplication (C = A * B)
    Author: Parikshit Gehlaut
    Memory Access: Streaming-like in naive implementation
*/
#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <map>
#include <cassert>
using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(_e) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device function to get ID of SM on which current thread is running
static __device__ __forceinline__ unsigned smid() {
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

// Structure to store the start clock, end clock, and SM ID for a single warp
struct WarpEvent {
    uint64_t start_clock;
    uint64_t end_clock;
    unsigned int sm_id;
};

__global__ void matmulNaive(const float *A, const float *B, float *C, int N,
                            WarpEvent* warp_events) {
    // Declare shared memory for Occupancy Control (dummy usage)
    extern __shared__ char dummy_shmem[];

    // Thread and block indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Occupancy measurement logic
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id_in_block = linear_tid / warpSize;
    int lane_id = linear_tid % warpSize;
    int warps_per_block = (blockDim.x * blockDim.y) / warpSize;
    int block_linear = blockIdx.y * gridDim.x + blockIdx.x;
    int warp_global_id = block_linear * warps_per_block + warp_id_in_block;

    if (lane_id == 0) {
        warp_events[warp_global_id].start_clock = clock64();
        warp_events[warp_global_id].sm_id = smid();
    }

    // Naive matrix multiplication
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

    if (lane_id == 0) {
        __threadfence_system();
        warp_events[warp_global_id].end_clock = clock64();
    }
}

// Host Code
int main(int argc, char* argv[]) {
    // Expect 4 command-line arguments: N, shmem_bytes, threads_per_block (ignores extras for bash compat)
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <N> <dummy_iters> <dummy_array_mb> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        return 1;
    }

    // Parse arguments (compat with original bash: use argv[1]=N, argv[4]=shmem, argv[5]=TPB; ignore others)
    int N = std::stoi(argv[1]);
    size_t shmem_bytes = static_cast<size_t>(std::stoll(argv[4]));
    int threads_per_block = std::stoi(argv[5]);

    // Validate
    if (N <= 0 || threads_per_block <= 0 || threads_per_block % 16 != 0) {
        cerr << "Invalid N or threads_per_block (must be positive multiple of 16)\n";
        return 1;
    }

    // GPU Information
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
         << ", warpSize: " << prop.warpSize << "\n";

    // Block dimensions: fixed y=16, x = threads_per_block / 16 (warp-aligned)
    int block_y = 16;
    int block_x = threads_per_block / block_y;
    if (block_x * block_y != threads_per_block) {
        cerr << "threads_per_block must be multiple of 16 for block_y=16\n";
        return 1;
    }
    if (N % block_y != 0 || N % block_x != 0) {
        cerr << "N must be multiple of block_x (" << block_x << ") and block_y (" << block_y << ")\n";
        return 1;
    }

    // Matrix setup
    size_t matrix_bytes = static_cast<size_t>(N) * N * sizeof(float);
    vector<float> h_A(N * N), h_B(N * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>((i % 101) - 50) / 10.0f;
        h_B[i] = static_cast<float>(((i + 7) % 103) - 50) / 10.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));
    CUDA_CHECK(cudaMemset(d_C, 0, matrix_bytes));  // Init C to 0
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));

    // Grid and block dims
    dim3 threadsPerBlock(block_x, block_y);
    dim3 numBlocks(N / block_x, N / block_y);

    // Total warps
    int warps_per_block = threads_per_block / prop.warpSize;
    size_t totalWarps = static_cast<size_t>(numBlocks.x) * numBlocks.y * warps_per_block;

    WarpEvent *d_warp_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_warp_events, totalWarps * sizeof(WarpEvent)));

    // CUDA events
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Set max dynamic shared memory
    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)matmulNaive, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));

    // Warmup
    CUDA_CHECK(cudaMemset(d_warp_events, 0, totalWarps * sizeof(WarpEvent)));
    matmulNaive<<<numBlocks, threadsPerBlock, shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaMemset(d_warp_events, 0, totalWarps * sizeof(WarpEvent)));
    CUDA_CHECK(cudaEventRecord(ev_start));
    matmulNaive<<<numBlocks, threadsPerBlock, shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    // Process occupancy
    vector<WarpEvent> h_events(totalWarps);
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, totalWarps * sizeof(WarpEvent), cudaMemcpyDeviceToHost));

    struct Timestamp {
        uint64_t time;
        int type;
        bool operator<(const Timestamp& o) const { return time < o.time; }
    };

    map<unsigned int, vector<Timestamp>> events_by_sm;
    for (size_t w = 0; w < totalWarps; ++w) {
        if (h_events[w].start_clock == 0 && h_events[w].end_clock == 0) continue;
        events_by_sm[h_events[w].sm_id].push_back({h_events[w].start_clock, +1});
        events_by_sm[h_events[w].sm_id].push_back({h_events[w].end_clock, -1});
    }

    int overall_max_occ = 0;
    for (auto& kv : events_by_sm) {
        if (kv.second.empty()) continue;
        sort(kv.second.begin(), kv.second.end());
        int current_occupancy = 0;
        int max_occupancy_this_sm = 0;
        for (auto& ts : kv.second) {
            current_occupancy += ts.type;
            if (current_occupancy > max_occupancy_this_sm) {
                max_occupancy_this_sm = current_occupancy;
            }
        }
        if (max_occupancy_this_sm > overall_max_occ) {
            overall_max_occ = max_occupancy_this_sm;
        }
    }

    // Compute metrics
    double time_s = static_cast<double>(milliseconds) / 1000.0;
    double total_flops = 2.0 * static_cast<double>(N) * N * N;  // 2 FLOPs per element (mul + add)
    double gflops = (time_s > 1e-9) ? (total_flops / (time_s * 1e9)) : 0.0;

    double total_bytes = 3.0 * static_cast<double>(N) * N * sizeof(float);  // A read, B read, C write
    double throughput_GBps = (time_s > 1e-9) ? (total_bytes / time_s / (1024.0 * 1024.0 * 1024.0)) : 0.0;

    // Output in parseable format
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "MatrixSize_N: " << N << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GFLOPS: " << std::fixed << gflops << "\n";
    cout << "Throughput_GBps: " << std::fixed << throughput_GBps << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << overall_max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_warp_events));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return 0;
}