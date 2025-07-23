#include <iostream>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include "evaluator.cuh"

__device__ void whoami(void) {
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

// Kernel to initialize cuRAND states
__global__ void setupRNG(curandState *state, unsigned long seed, int num_threads)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_threads) {
        // Each thread gets different sequence number for better randomness
        curand_init(seed, tid, 0, &state[tid]);
    }
}

// Kernel to deal and evaluate poker hands
__global__ void dealAndEvaluateHands(Hand* d_hands, int* d_results, int num_hands, curandState* rng_states)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // whoami();
    
    if (tid < num_hands) {
        PokerHand pokerHand;
        
        // Deal a hand for this thread using cuRAND
        pokerHand.dealHand(d_hands, tid, rng_states);
        
        // Evaluate the hand
        d_results[tid] = EvaluatorHand{}(d_hands[tid]);
    }
}

// Host function to print a single hand nicely
void printHand(const Hand& hand) {
    uint8_t cards[5] = {
        thrust::get<0>(hand), thrust::get<1>(hand), thrust::get<2>(hand),
        thrust::get<3>(hand), thrust::get<4>(hand)
    };

    EvaluatorHelper helper;
    helper.bubbleSortHand(cards);  // host call!
    
    for (int i = 0; i < 5; i++) {
        int rank = cards[i] % 13;
        int suit = cards[i] / 13;
        std::cout << "23456789TJQKA"[rank] << "CDHS"[suit];
        if (i < 4) std::cout << " ";
    }
}

// Host function to print hand type statistics
void printStatistics(const thrust::host_vector<int>& results, const thrust::host_vector<Hand>& hands, int num_hands)
{
    const char* handTypeNames[] = {
        "High Card", "One Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind",
        "Straight Flush", "Royal Flush"
    };
    
    std::vector<int> counts(10, 0);
    std::vector<std::vector<Hand>> handExamples(10);
    
    // Count each hand type and collect examples
    for (int i = 0; i < num_hands; i++) {
        int result = results[i];
        if (result >= 0 && result < 10) {
            counts[result]++;
            // Store first few examples of rare hands
            if (result >= 6 && handExamples[result].size() < 5) {
                handExamples[result].push_back(hands[i]);
            }
        }
    }
    
    std::cout << "\n=== Poker Hand Statistics (" << num_hands << " hands) ===\n";
    for (int i = 0; i < 10; i++) {
        double percentage = (double)counts[i] / num_hands * 100.0;
        std::cout << handTypeNames[i] << ": " << counts[i] 
                  << " (" << std::fixed << std::setprecision(4) << percentage << "%)";
        
        // Show examples of rare hands (Full House and above)
        if (i >= 6 && !handExamples[i].empty()) {
            std::cout << "\n  Examples: ";
            for (size_t j = 0; j < handExamples[i].size(); j++) {
                if (j > 0) std::cout << ", ";
                printHand(handExamples[i][j]);
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main() {
    #ifdef NDEBUG
        std::cout << "Release configuration!\n";
    #else
        std::cout << "Debug configuration!\n";
    #endif

    const int NUM_HANDS = 2598960 * 2;  // (52/5) permutations
    // const int NUM_HANDS = 1025;  // (52/5) permutations
    // const int THREADS_PER_BLOCK = 256;
    const int THREADS_PER_BLOCK = 512;
    const int NUM_BLOCKS = (NUM_HANDS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    std::cout << "Dealing and evaluating " << NUM_HANDS << " poker hands on GPU...\n";
    std::cout << "Using " << NUM_BLOCKS << " blocks with " << THREADS_PER_BLOCK << " threads each\n";
    
    // Generate seed from current time
    unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    // Allocate device memory using thrust::device_vector (RAII approach)
    thrust::device_vector<Hand> d_hands(NUM_HANDS);
    thrust::device_vector<int> d_results(NUM_HANDS);
    
    // Allocate cuRAND states for each thread
    curandState* d_rng_states;
    cudaMalloc(&d_rng_states, NUM_HANDS * sizeof(curandState));
    
    std::cout << "Initializing random number generators...\n";
    
    // Initialize cuRAND states
    setupRNG<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_rng_states, seed, NUM_HANDS);
    cudaDeviceSynchronize();
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    dealAndEvaluateHands<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_hands.data()),
        thrust::raw_pointer_cast(d_results.data()),
        NUM_HANDS,
        d_rng_states
    );
    
    // Wait for GPU to finish
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }
    
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy results and hands back to host
    thrust::host_vector<int> h_results = d_results;
    thrust::host_vector<Hand> h_hands = d_hands;
    
    // std::cout << "\nGPU processing completed in " << duration.count() << " ms\n";
    // std::cout << "Performance: " << (NUM_HANDS / (duration.count() / 1000.0)) << " hands/second\n";

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    double ms = duration_ns.count() / 1000000.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nGPU processing completed in " << ms << " ms\n";
    std::cout << "Performance: " << (NUM_HANDS / (ms / 1000.0)) << " hands/second\n";
    
    // Print statistics with examples of rare hands
    printStatistics(h_results, h_hands, NUM_HANDS);
    
    // Cleanup cuRAND states
    cudaFree(d_rng_states);
    
    return EXIT_SUCCESS;
}