/*
    Workload: Tiled Matrix Multiplication (C = A * B)
    Features: Flexible Threads-Per-Block (TPB) for Volkov Analysis
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include <cstdint>
#include <cassert>
#include <string>

using namespace std;

#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(_e) << std::endl; \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// SM ID retrieval
static __device__ __forceinline__ unsigned smid() {
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

struct WarpEvent {
    uint64_t start_clock;
    uint64_t end_clock;
    unsigned int sm_id;
};

// Fixed Maximum array size for local register tiling
// T=16 (256 elements), Min TPB=32 => Max 8 elements per thread. 
// We set 16 to be safe.
#define MAX_ELEMS_PER_THREAD 16

template <int T>
__global__ void matmulTiledGeneral(const float *A, const float *B, float *C, int N,
                                   WarpEvent* warp_events) {
    // 1. Shared Memory for Tile (Static)
    __shared__ float tileA[T][T];
    __shared__ float tileB[T][T];

    // 2. Dummy Shared Memory (Dynamic) for Occupancy Control
    extern __shared__ char dummy_shmem[];
    if (threadIdx.x == 0 && dummy_shmem[0] == 1) { /* prevent optimization */ }

    // IDs
    int tid = threadIdx.x;
    int block_size = blockDim.x; // TPB
    
    // Block offsets
    int blockRow = blockIdx.y * T;
    int blockCol = blockIdx.x * T;

    // -- Measurement Logic --
    int warps_per_block = block_size / warpSize;
    if (warps_per_block < 1) warps_per_block = 1;
    
    int warp_id_in_block = tid / warpSize;
    int lane_id = tid % warpSize;
    int block_linear = blockIdx.y * gridDim.x + blockIdx.x;
    int warp_global_id = block_linear * warps_per_block + warp_id_in_block;

    if (lane_id == 0 && warp_id_in_block < warps_per_block) {
        warp_events[warp_global_id].start_clock = clock64();
        warp_events[warp_global_id].sm_id = smid();
    }

    // -- Local Accumulators --
    // Each thread handles specific pixels in the TxT tile. 
    // We pre-calculate which indices this thread owns to avoid re-calculating logic inside the hot loop.
    float acc[MAX_ELEMS_PER_THREAD] = {0.0f};

    int numTiles = N / T;

    // -- Main Loop Over Tiles --
    for (int m = 0; m < numTiles; ++m) {
        
        // 1. Collaborative Load into Shared Memory
        // The TxT tile has T*T elements. We iterate over them with stride = block_size
        for (int i = tid; i < T * T; i += block_size) {
            int r = i / T;
            int c = i % T;
            
            int globalRowA = blockRow + r;
            int globalColA = m * T + c;
            
            int globalRowB = m * T + r;
            int globalColB = blockCol + c;

            // Padding/Bound check
            tileA[r][c] = (globalRowA < N && globalColA < N) ? A[globalRowA * N + globalColA] : 0.0f;
            tileB[r][c] = (globalRowB < N && globalColB < N) ? B[globalRowB * N + globalColB] : 0.0f;
        }

        __syncthreads();

        // 2. Compute
        // Each thread updates its own accumulators
        int idx = 0;
        for (int i = tid; i < T * T; i += block_size) {
            int r = i / T;
            int c = i % T;
            
            float partial = 0.0f;
            
            #pragma unroll
            for (int k = 0; k < T; ++k) {
                partial += tileA[r][k] * tileB[k][c];
            }
            
            acc[idx] += partial;
            idx++;
        }

        __syncthreads();
    }

    // 3. Write Results to Global Memory
    int idx = 0;
    for (int i = tid; i < T * T; i += block_size) {
        int r = i / T;
        int c = i % T;
        
        int globalRow = blockRow + r;
        int globalCol = blockCol + c;
        
        if (globalRow < N && globalCol < N) {
            C[globalRow * N + globalCol] = acc[idx];
        }
        idx++;
    }

    // -- Measurement End --
    if (lane_id == 0 && warp_id_in_block < warps_per_block) {
        __threadfence_system();
        warp_events[warp_global_id].end_clock = clock64();
    }
}

int main(int argc, char* argv[]) {
    // Arguments: N, iters, dummy_mb, shmem_bytes, tile_dim, num_blocks, threads_per_block
    if (argc < 8) {
        cerr << "Usage: " << argv[0] << " <N> <iters> <dummy_mb> <shmem_bytes> <TILE_DIM> <num_blocks> <TPB>\n";
        return 1;
    }

    int N = std::stoi(argv[1]);
    size_t user_shmem_bytes = static_cast<size_t>(std::stoll(argv[4]));
    int tile_dim = std::stoi(argv[5]);
    // argv[6] is num_blocks (ignored)
    int threads_per_block = std::stoi(argv[7]);

    if (N % tile_dim != 0) {
        cerr << "Error: N must be a multiple of TILE_DIM.\n";
        return 1;
    }

    // Validate TPB sufficiency
    int total_pixels = tile_dim * tile_dim;
    int pixels_per_thread = (total_pixels + threads_per_block - 1) / threads_per_block;
    if (pixels_per_thread > MAX_ELEMS_PER_THREAD) {
        cerr << "Error: TPB=" << threads_per_block << " too small for Tile=" << tile_dim 
             << ". Requires " << pixels_per_thread << " regs (Max " << MAX_ELEMS_PER_THREAD << ")\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Alloc
    size_t matrix_bytes = static_cast<size_t>(N) * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, matrix_bytes));
    
    // Init data (dummy)
    CUDA_CHECK(cudaMemset(d_A, 1, matrix_bytes)); 
    CUDA_CHECK(cudaMemset(d_B, 1, matrix_bytes));

    // Config
    dim3 block(threads_per_block);
    dim3 grid(N / tile_dim, N / tile_dim);
    
    int warps_per_block = threads_per_block / prop.warpSize;
    if (warps_per_block == 0) warps_per_block = 1;
    size_t totalWarps = static_cast<size_t>(grid.x) * grid.y * warps_per_block;

    WarpEvent *d_warp_events;
    CUDA_CHECK(cudaMalloc(&d_warp_events, totalWarps * sizeof(WarpEvent)));
    CUDA_CHECK(cudaMemset(d_warp_events, 0, totalWarps * sizeof(WarpEvent)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Shared Memory Calculation
    // MaxOptin is the hardware ceiling.
    int max_hardware_shmem_optin;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_hardware_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    
    // We need 'user_shmem_bytes' (Dynamic) + 'tileA + tileB' (Static)
    // to be <= max_hardware_shmem_optin.
    // The cudaFuncSetAttribute sets the limit for DYNAMIC.
    int static_shmem_bytes = 2 * tile_dim * tile_dim * sizeof(float);
    int max_dynamic = max_hardware_shmem_optin - static_shmem_bytes;
    
    if (max_dynamic < 0) {
        cerr << "Error: Static shared memory exceeds hardware limit.\n";
        return 1;
    }
    
    // If user requests more than possible, clamp or fail? The kernel launch will fail if we don't enable enough.
    // But we can only enable up to max_dynamic.
    if (user_shmem_bytes > (size_t)max_dynamic) {
        // We cannot launch this configuration on this hardware
        cerr << "Error: Requested Shmem (" << user_shmem_bytes << ") + Static (" 
             << static_shmem_bytes << ") > HW Limit (" << max_hardware_shmem_optin << ")\n";
        return 1;
    }

    // Launch Logic
    void* kernel_func = nullptr;
    if (tile_dim == 8) kernel_func = (void*)matmulTiledGeneral<8>;
    else if (tile_dim == 16) kernel_func = (void*)matmulTiledGeneral<16>;
    else { cerr << "Unsupported T=" << tile_dim << endl; return 1; }

    // 1. Set Attribute
    CUDA_CHECK(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic));

    // 2. Warmup
    if (tile_dim == 8) matmulTiledGeneral<8><<<grid, block, user_shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);
    else matmulTiledGeneral<16><<<grid, block, user_shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Warmup Launch Failed: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Timed Run
    CUDA_CHECK(cudaMemset(d_warp_events, 0, totalWarps * sizeof(WarpEvent)));
    CUDA_CHECK(cudaEventRecord(start));
    
    if (tile_dim == 8) matmulTiledGeneral<8><<<grid, block, user_shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);
    else matmulTiledGeneral<16><<<grid, block, user_shmem_bytes>>>(d_A, d_B, d_C, N, d_warp_events);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Process Occupancy
    vector<WarpEvent> h_events(totalWarps);
    CUDA_CHECK(cudaMemcpy(h_events.data(), d_warp_events, totalWarps * sizeof(WarpEvent), cudaMemcpyDeviceToHost));

    map<unsigned int, vector<pair<uint64_t, int>>> events_by_sm;
    for (const auto& e : h_events) {
        if (e.start_clock == 0) continue;
        events_by_sm[e.sm_id].push_back({e.start_clock, 1});
        events_by_sm[e.sm_id].push_back({e.end_clock, -1});
    }

    int max_occ = 0;
    for (auto& kv : events_by_sm) {
        auto& timeline = kv.second;
        sort(timeline.begin(), timeline.end());
        int curr = 0, local_max = 0;
        for (auto& t : timeline) {
            curr += t.second;
            if (curr > local_max) local_max = curr;
        }
        if (local_max > max_occ) max_occ = local_max;
    }

    double time_s = milliseconds / 1000.0;
    double gflops = (2.0 * N * N * N) / (time_s * 1e9);
    double total_bytes = 3.0 * static_cast<double>(N) * N * sizeof(float);
    double gbps = total_bytes / time_s / 1e9;

    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "MatrixSize_N: " << N << "\n";
    cout << "TileSize: " << tile_dim << "\n";
    cout << "ExecutionTime_ms: " << milliseconds << "\n";
    cout << "Throughput_GFLOPS: " << gflops << "\n";
    cout << "Throughput_GBps: " << gbps << "\n";
    cout << "MaximumAttainedOccupancy_warpsPerSM: " << max_occ << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    return 0;
}