/*
    Workload : Arithmetic Latency Measurement (Based on Volkov)
    Author   : Parikshit Gehlaut
    Access   : Register-to-Register (No Global/Shared Memory)
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
// This is identical to the memory latency version
struct WarpResult {
    unsigned int start_clock;
    unsigned int end_clock;
    unsigned int sm_id;
    uintptr_t sink; // To prevent compiler optimization
};

// This factor amortizes the outer loop overhead.
// Total ops = outer_iterations * ARITH_UNROLL_FACTOR
const int ARITH_UNROLL_FACTOR = 32;

/*
    Kernel to measure FADD (FP32 Add) latency.
    - Creates a dependent chain of FADD operations.
    - `c = c + a`
*/
__global__ void fadd_latency_kernel(int outer_iterations,
                                    WarpResult* warp_events)
{
    // Shared memory is allocated by the launcher to control occupancy
    extern __shared__ int shmem[]; 
    
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int block_linear_id = blockIdx.x;
    int warp_global_id = block_linear_id * warps_per_block + warp_id_in_block;

    // Registers for the arithmetic chain
    // Using volatile to encourage compiler to keep them in registers
    // and not optimize them to constants.
    volatile float a = 1.000001f;
    volatile float c = (float)lane_id * 0.1f; // Accumulator

    // Lane 0 records start time and SM ID
    if(lane_id == 0){
        warp_events[warp_global_id].sm_id = smid();
        __threadfence_block(); 
        warp_events[warp_global_id].start_clock = clock();
    }

    // The core dependent arithmetic loop
    for(int i=0; i<outer_iterations; i++){
        
        // This inner loop is unrolled.
        // The dependency on 'c' forces serial execution.
        #pragma unroll ARITH_UNROLL_FACTOR
        for(int j=0; j < ARITH_UNROLL_FACTOR; j++) {
            c = c + a;
        }
    }
    
    // Warp-wide reduction to sink *all* thread results
    // This prevents the compiler from optimizing away the entire loop.
    unsigned long long c_ll = (unsigned long long)__float_as_int(c);
    #pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        c_ll += __shfl_down_sync(0xFFFFFFFF, c_ll, offset);
    }

    // Lane 0 records end time and writes the final summed sink
    if (lane_id == 0) {
        warp_events[warp_global_id].end_clock = clock();
        warp_events[warp_global_id].sink = (uintptr_t)c_ll;
        shmem[0] = (int)c_ll; // Write to shared mem as an extra sink
    }
}


/*
    Kernel to measure FMUL (FP32 Multiply) latency.
    - Creates a dependent chain of FMUL operations.
    - `c = c * a`
*/
__global__ void fmul_latency_kernel(int outer_iterations,
                                    WarpResult* warp_events)
{
    extern __shared__ int shmem[]; 
    
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int block_linear_id = blockIdx.x;
    int warp_global_id = block_linear_id * warps_per_block + warp_id_in_block;

    volatile float a = 1.000001f; // Use a value != 1.0 to avoid optimization
    volatile float c = 1.0f + (float)lane_id * 0.01f; 

    if(lane_id == 0){
        warp_events[warp_global_id].sm_id = smid();
        __threadfence_block(); 
        warp_events[warp_global_id].start_clock = clock();
    }

    for(int i=0; i<outer_iterations; i++){
        #pragma unroll ARITH_UNROLL_FACTOR
        for(int j=0; j < ARITH_UNROLL_FACTOR; j++) {
            c = c * a;
        }
    }
    
    unsigned long long c_ll = (unsigned long long)__float_as_int(c);
    #pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        c_ll += __shfl_down_sync(0xFFFFFFFF, c_ll, offset);
    }

    if (lane_id == 0) {
        warp_events[warp_global_id].end_clock = clock();
        warp_events[warp_global_id].sink = (uintptr_t)c_ll;
        shmem[0] = (int)c_ll;
    }
}


/*
    Kernel to measure FMA (Fused Multiply-Add) latency.
    - Creates a dependent chain of FMA operations.
    - `c = a * b + c`
*/
__global__ void fma_latency_kernel(int outer_iterations,
                                   WarpResult* warp_events)
{
    extern __shared__ int shmem[]; 
    
    int warp_id_in_block = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warps_per_block = blockDim.x / warpSize;
    int block_linear_id = blockIdx.x;
    int warp_global_id = block_linear_id * warps_per_block + warp_id_in_block;

    volatile float a = 1.000001f;
    volatile float b = 0.999999f;
    volatile float c = (float)lane_id * 0.1f; // Accumulator

    if(lane_id == 0){
        warp_events[warp_global_id].sm_id = smid();
        __threadfence_block(); 
        warp_events[warp_global_id].start_clock = clock();
    }

    for(int i=0; i<outer_iterations; i++){
        #pragma unroll ARITH_UNROLL_FACTOR
        for(int j=0; j < ARITH_UNROLL_FACTOR; j++) {
            // Use intrinsic for precise FMA
            c = __fmaf_rn(a, b, c); 
        }
    }
    
    unsigned long long c_ll = (unsigned long long)__float_as_int(c);
    #pragma unroll
    for(int offset = warpSize/2; offset > 0; offset /= 2) {
        c_ll += __shfl_down_sync(0xFFFFFFFF, c_ll, offset);
    }

    if (lane_id == 0) {
        warp_events[warp_global_id].end_clock = clock();
        warp_events[warp_global_id].sink = (uintptr_t)c_ll;
        shmem[0] = (int)c_ll;
    }
}


// Host Code
int main(int argc, char* argv[]) {
    // --- Argument Parsing ---
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <op_type> <outer_iterations> <shmem_bytes_per_block> <threads_per_block> <num_blocks>\n";
        cerr << "   op_type: 1=FADD, 2=FMUL, 3=FMA\n";
        return 1;
    }
    int op_type = std::stoi(argv[1]);
    int outer_iterations = std::stoi(argv[2]); 
    size_t shmem_bytes = static_cast<size_t>(std::stoll(argv[3]));
    int threads_per_block = std::stoi(argv[4]);
    int num_blocks = std::stoi(argv[5]);

    string op_name;
    switch(op_type) {
        case 1: op_name = "FADD"; break;
        case 2: op_name = "FMUL"; break;
        case 3: op_name = "FMA"; break;
        default:
            cerr << "Invalid op_type: " << op_type << ". Must be 1, 2, or 3." << endl;
            return 1;
    }

    // --- GPU Information ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << ", SMs: " << prop.multiProcessorCount
              << ", warpSize: " << prop.warpSize << ", Clock: " << (prop.clockRate / 1e6) << " GHz\n";
    assert((threads_per_block % prop.warpSize) == 0 && "threads per block must be multiple of warpSize");

    // --- NO MORE POINTER CHASING ARRAY ---

    // --- Kernel Launch Setup ---
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocksGrid(num_blocks);
    int warps_per_block = threads_per_block / prop.warpSize;
    int totalWarps = num_blocks * warps_per_block;

    WarpResult *d_warp_events = nullptr;
    CUDA_CHECK(cudaMalloc(&d_warp_events, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));

    // --- Set Shared Memory Attribute for ALL kernels ---
    int max_shmem_bytes;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shmem_bytes,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    
    CUDA_CHECK(cudaFuncSetAttribute(fadd_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(fmul_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(fma_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes));


    cout << "Launching " << op_name << " kernel with outer_iterations=" << outer_iterations 
              << ", shmem=" << shmem_bytes << "..." << endl;

    // --- Warmup Launch (1 iteration) ---
    switch(op_type) {
        case 1: fadd_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(1, d_warp_events); break;
        case 2: fmul_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(1, d_warp_events); break;
        case 3: fma_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(1, d_warp_events); break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed Kernel Launch ---
    CUDA_CHECK(cudaMemset(d_warp_events, 0, static_cast<size_t>(totalWarps) * sizeof(WarpResult)));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    switch(op_type) {
        case 1: fadd_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(outer_iterations, d_warp_events); break;
        case 2: fmul_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(outer_iterations, d_warp_events); break;
        case 3: fma_latency_kernel<<<numBlocksGrid, threadsPerBlock, shmem_bytes>>>(outer_iterations, d_warp_events); break;
    }
        
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
    
    // Total operations per warp = outer_iterations * ARITH_UNROLL_FACTOR
    const double total_ops_per_warp = (double)outer_iterations * ARITH_UNROLL_FACTOR;

    for (auto const& [smid, results_vec] : sm_results) {
        if (results_vec.empty()) continue;

        for (const auto& res : results_vec) {
            uint64_t duration = (uint64_t)res.end_clock - (uint64_t)res.start_clock;
            
            if (res.end_clock < res.start_clock) {
                duration += CLOCK_WRAP;
            }
            
            global_total_cycles += duration;
            global_total_warps++;
            
            // Normalize by total operations
            double current_latency_op = (double)duration / total_ops_per_warp;
            if (current_latency_op < global_min_latency_per_op) {
                global_min_latency_per_op = current_latency_op;
            }
        }
    }

    double global_mean_latency = 0;
    if (global_total_warps > 0 && total_ops_per_warp > 0) {
        double avg_cycles_per_warp = (double)global_total_cycles / global_total_warps;
        // Normalize by total operations
        global_mean_latency = avg_cycles_per_warp / total_ops_per_warp;
    } else {
        global_min_latency_per_op = 0;
    }

    // --- Print Summary ---
    cout << "\n===RESULT_SUMMARY_START===\n";
    cout << "OpType: " << op_name << "\n";
    cout << "OuterIterations: " << outer_iterations << "\n";
    cout << "TotalOpsPerWarp: " << (int)total_ops_per_warp << "\n";
    cout << "Shmem_Bytes: " << shmem_bytes << "\n";
    cout << "ThreadsPerBlock: " << threads_per_block << "\n";
    cout << "NumBlocks: " << num_blocks << "\n";
    cout << "TotalWarpsMeasured: " << global_total_warps << "\n";
    cout << "MeanLatency_cycles: " << std::fixed << std::setprecision(2) << global_mean_latency << "\n";
    cout << "MinLatency_cycles: " << std::fixed << std::setprecision(2) << global_min_latency_per_op << "\n";
    cout << "===RESULT_SUMMARY_END===\n";

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_warp_events));

    return 0;
}