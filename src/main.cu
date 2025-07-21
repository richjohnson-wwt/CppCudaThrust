#include <iostream>
#include <chrono>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include "poker.cuh"
#include "evaluator.cuh"

// Kernel to deal and evaluate poker hands
__global__ void dealAndEvaluateHands(Hand* d_hands, int* d_results, int num_hands, int seed)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < num_hands) {
        evaluate_hand evaluator;
        
        // Deal a hand for this thread
        evaluator.dealHand(d_hands, tid, seed);
        
        // Evaluate the hand
        d_results[tid] = evaluator(d_hands[tid]);
    }
}

// Host function to print a single hand nicely
void printHand(const Hand& hand) {
    int cards[5] = {
        thrust::get<0>(hand), thrust::get<1>(hand), thrust::get<2>(hand),
        thrust::get<3>(hand), thrust::get<4>(hand)
    };
    
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

    const int NUM_HANDS = 2598960;  // (52/5) permutations
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (NUM_HANDS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    std::cout << "Dealing and evaluating " << NUM_HANDS << " poker hands on GPU...\n";
    std::cout << "Using " << NUM_BLOCKS << " blocks with " << THREADS_PER_BLOCK << " threads each\n";
    
    // Generate seed from current time
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    // Allocate device memory using thrust::device_vector (RAII approach)
    thrust::device_vector<Hand> d_hands(NUM_HANDS);
    thrust::device_vector<int> d_results(NUM_HANDS);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    dealAndEvaluateHands<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_hands.data()),
        thrust::raw_pointer_cast(d_results.data()),
        NUM_HANDS,
        seed
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
    
    std::cout << "\nGPU processing completed in " << duration.count() << " ms\n";
    std::cout << "Performance: " << (NUM_HANDS / (duration.count() / 1000.0)) << " hands/second\n";
    
    // Print statistics with examples of rare hands
    printStatistics(h_results, h_hands, NUM_HANDS);
    
    return EXIT_SUCCESS;
}