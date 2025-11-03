#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <limits>

using namespace std;

// Macro to check for CUDA errors
#define CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Struct to hold the timing results for each warp
struct WarpResult {
    unsigned int start_time;
    unsigned int end_time;
    unsigned int sm_id;
    unsigned int sink; // To prevent the compiler from optimizing out the loop
};

__global__ void latency_kernel(const unsigned int* __restrict__ next_ptr,
                               WarpResult* results,
                               const int num_ops)
{
    const unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const unsigned int lane_id = threadIdx.x % warpSize;

    // Only one thread per warp (the first one) does the work
    if (lane_id == 0) {
        unsigned int smid;
        asm volatile ("mov.u32 %0, %%smid;" : "=r"(smid));

        // Each warp starts at a unique index
        unsigned int current_idx = warp_id;

        // Memory fence to ensure all previous memory operations are complete before timing
        asm volatile("membar.cta;");
        unsigned int start_clk;
        asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_clk));

        // The core pointer-chasing loop
        #pragma unroll 1 // Prevent unrolling to ensure dependent loads
        for (int i = 0; i < num_ops; ++i) {
            current_idx = next_ptr[current_idx];
        }
        
        unsigned int end_clk;
        asm volatile ("mov.u32 %0, %%clock;" : "=r"(end_clk));
        // Memory fence to ensure the timing captures the memory operations
        asm volatile("membar.cta;");

        // Store the results
        results[warp_id].start_time = start_clk;
        results[warp_id].end_time = end_clk;
        results[warp_id].sm_id = smid;
        results[warp_id].sink = current_idx; // This makes the loop's result essential
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d), %.3f GHz\n\n", prop.name, prop.major, prop.minor, prop.clockRate / 1e6);

    const size_t mem_size_bytes = 512 * 1024 * 1024;
    const size_t num_elements = mem_size_bytes / sizeof(unsigned int);
    const int mem_ops_per_warp = 100000;
    
    printf("Setting up %zu MB pointer-chasing array...\n", mem_size_bytes / (1024*1024));
    vector<unsigned int> h_indices(num_elements);
    iota(h_indices.begin(), h_indices.end(), 0);

    // Shuffle the indices to create a random access pattern
    mt19937 rng(random_device{}());
    shuffle(h_indices.begin(), h_indices.end(), rng);
    
    // Create the pointer-chasing list from the shuffled indices
    std::vector<unsigned int> h_next_ptr(num_elements);
    for (size_t i = 0; i < num_elements - 1; ++i) {
        h_next_ptr[h_indices[i]] = h_indices[i + 1];
    }
    // Make the list circular
    h_next_ptr[h_indices.back()] = h_indices[0];

    unsigned int* d_next_ptr;
    WarpResult* d_results; // Declare pointer
    CHECK(cudaMalloc(&d_next_ptr, mem_size_bytes));
    CHECK(cudaMemcpy(d_next_ptr, h_next_ptr.data(), mem_size_bytes, cudaMemcpyHostToDevice));

    double min_latency = std::numeric_limits<double>::max();

    printf("Performing a warm-up run...\n");
    // The warm-up launch uses 1 block of 32 threads, which is 1 warp.
    CHECK(cudaMalloc(&d_results, 1 * sizeof(WarpResult)));
    
    latency_kernel<<<1, 32>>>(d_next_ptr, d_results, 1);
    CHECK(cudaGetLastError()); // Good practice to check after kernel launch
    CHECK(cudaDeviceSynchronize());

    // Free the memory allocated for the warm-up run.
    CHECK(cudaFree(d_results));

    printf("Running benchmark across different occupancies...\n");
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    for (int warps_per_sm = 1; warps_per_sm <= max_warps_per_sm; warps_per_sm++) {
        int num_warps = prop.multiProcessorCount * warps_per_sm;
        int num_threads = num_warps * prop.warpSize;
        dim3 threads_per_block(256, 1, 1);
        dim3 num_blocks((num_threads + threads_per_block.x - 1) / threads_per_block.x, 1, 1);

        CHECK(cudaMalloc(&d_results, num_warps * sizeof(WarpResult)));

        latency_kernel<<<num_blocks, threads_per_block>>>(d_next_ptr, d_results, mem_ops_per_warp);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        
        std::vector<WarpResult> h_results(num_warps);
        CHECK(cudaMemcpy(h_results.data(), d_results, num_warps * sizeof(WarpResult), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_results));

        std::map<unsigned int, std::vector<WarpResult>> sm_results;
        for (const auto& res : h_results) {
            if (res.sm_id < (unsigned int)prop.multiProcessorCount) {
                 sm_results[res.sm_id].push_back(res);
            }
        }
        
        double total_avg_latency = 0;
        int sm_count = 0;

        for (auto const& [smid, results_vec] : sm_results) {
            if (results_vec.empty()) continue;

            long long total_cycles_on_sm = 0;
            for (const auto& res : results_vec) {
                // Handle clock counter wrap-around
                long long duration = (long long)res.end_time - (long long)res.start_time;
                if (duration < 0) {
                    duration += (1LL << 32);
                }
                total_cycles_on_sm += duration;
            }
            total_avg_latency += (double)total_cycles_on_sm / results_vec.size();
            sm_count++;
        }

        if (sm_count > 0) {
            double mean_warp_latency = total_avg_latency / sm_count;
            double latency_per_op = mean_warp_latency / mem_ops_per_warp;
            min_latency = std::min(min_latency, latency_per_op);
        }
    }

    printf("\n--- RESULTS ---\n");
    printf("Unloaded Global Memory Latency: %.2f cycles\n", min_latency);

    CHECK(cudaFree(d_next_ptr));
    return 0;
}