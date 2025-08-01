#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../src/evaluator.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            FAIL("CUDA Error"); \
        } \
    } while (0)

__global__ void testUnpackTupleKernel(uint8_t* resultCards, uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) 
{
    EvaluatorHelper testObject; // Create an instance of evaluate_hand
    Hand hand = thrust::make_tuple(a, b, c, d, e);
    testObject.unpackTuple(hand, resultCards); // Call unpackTuple
}

__global__ void testRankAndSuitsKernel(uint8_t* resultRanks, uint8_t* resultSuits, uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) 
{
    EvaluatorHelper testObject; // Create an instance of evaluate_hand
    uint8_t hand[5] = {a, b, c, d, e};
    testObject.rankAndSuits(&hand[0], resultRanks, resultSuits);
}

__global__ void testBubbleSortHandKernel(uint8_t* resultCards) {
    EvaluatorHelper testObject; // Create an instance of evaluate_hand
    testObject.bubbleSortHand(resultCards);
    // Note: you can print from the GPU but not with std::cout
    printf("\n\nResult: %d, %d, %d, %d, %d\n\n", resultCards[0], resultCards[1], resultCards[2], resultCards[3], resultCards[4]);
}

__global__ void testRankAndPairCountsKernel(int* rankCount, int* pairCount, uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e)
{
    EvaluatorHelper testObject; // Create an instance of evaluate_hand
    uint8_t ranks[5] = {a, b, c, d, e};
    testObject.doRankAndPairCounts(ranks, *rankCount, *pairCount);
}

__global__ void testFlushAndStaightKernel(bool* resultIsFlush, bool* resultIsStraight, uint8_t* ranks, uint8_t* suits) 
{
    EvaluatorHelper testObject; // Create an instance of evaluate_hand
    testObject.doFlushAndStaight(&ranks[0], &suits[0], *resultIsFlush, *resultIsStraight);
}

__global__ void testOperatorOverloadKernel(int* resultScore, uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) 
{
    Hand hand = thrust::make_tuple(a, b, c, d, e);
    *resultScore = EvaluatorHand{}(hand);
}

__global__ void testEvaluateAllHandsKernel(Hand* d_hands_ptr, int* d_results_ptr, int count) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count) {
        d_results_ptr[i] = EvaluatorHand{}(d_hands_ptr[i]);
    }
}

__global__ void generateHandsKernel(Hand* d_hands, int num_hands, curandState* rng_states)
{
    PokerHand testObject;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_hands) {
        testObject.dealHand(d_hands, tid, rng_states);
    }
}

TEST_CASE("EvaluatorHelper Tests - unpackTuple", "[cuda][kernel][EvaluatorHelper]") 
{
    uint8_t host_cards[5] = {0, 0, 0, 0, 0};

    uint8_t* device_cards;

    CUDA_CHECK(cudaMalloc(&device_cards, sizeof(uint8_t) * 5));

    testUnpackTupleKernel<<<1, 1>>>(device_cards, 6, 15, 48, 13, 3);

    cudaDeviceSynchronize();

    cudaMemcpy(host_cards, device_cards, sizeof(uint8_t) * 5, cudaMemcpyDeviceToHost);
    cudaFree(device_cards);

    REQUIRE(host_cards[0] == 6);
    REQUIRE(host_cards[1] == 15);
    REQUIRE(host_cards[2] == 48);
    REQUIRE(host_cards[3] == 13);
    REQUIRE(host_cards[4] == 3);
}

TEST_CASE("EvaluatorHelper Tests - rankAndSuits", "[cuda][kernel][EvaluatorHelper]") 
{
    uint8_t host_ranks[5] = {0, 0, 0, 0, 0};
    uint8_t host_suits[5] = {0, 0, 0, 0, 0};

    uint8_t* device_ranks;
    uint8_t* device_suits;

    cudaMalloc(&device_ranks, sizeof(uint8_t) * 5);
    cudaMalloc(&device_suits, sizeof(uint8_t) * 5);

    testRankAndSuitsKernel<<<1, 1>>>(device_ranks, device_suits, 6, 15, 48, 13, 3);

    cudaDeviceSynchronize();

    cudaMemcpy(host_ranks, device_ranks, sizeof(uint8_t) * 5, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_suits, device_suits, sizeof(uint8_t) * 5, cudaMemcpyDeviceToHost);
    cudaFree(device_ranks);
    cudaFree(device_suits);

    REQUIRE(host_ranks[0] == 6);
    REQUIRE(host_ranks[1] == 2);
    REQUIRE(host_ranks[2] == 9);
    REQUIRE(host_ranks[3] == 0);
    REQUIRE(host_ranks[4] == 3);

    REQUIRE(host_suits[0] == 0);
    REQUIRE(host_suits[1] == 1);
    REQUIRE(host_suits[2] == 3);
    REQUIRE(host_suits[3] == 1);
    REQUIRE(host_suits[4] == 0);
}

TEST_CASE("EvaluatorHelper Test - bubbleSortHand", "[cuda][kernel][EvaluatorHelper]") 
{
    uint8_t host_cards[5] = {6, 2, 9, 0, 3};
    uint8_t host_result[5] = {0, 0, 0, 0, 0};

    uint8_t* device_cards;

    cudaMalloc(&device_cards, sizeof(uint8_t) * 5);

    cudaMemcpy(device_cards, host_cards, sizeof(uint8_t) * 5, cudaMemcpyHostToDevice);

    testBubbleSortHandKernel<<<1, 1>>>(device_cards);

    cudaDeviceSynchronize();

    cudaMemcpy(host_result, device_cards, sizeof(uint8_t) * 5, cudaMemcpyDeviceToHost);
    cudaFree(device_cards);

    REQUIRE(host_result[0] == 0);
    REQUIRE(host_result[1] == 2);
    REQUIRE(host_result[2] == 3);
    REQUIRE(host_result[3] == 6);
    REQUIRE(host_result[4] == 9);
}

TEST_CASE("EvaluatorHelper Test - doRankAndPairCounts", "[cuda][kernel][EvaluatorHelper]") 
{
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    SECTION("HighCard")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 6, 2, 9, 0, 3);

        cudaDeviceSynchronize();
    
        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);
    
        REQUIRE(h_maxRankCount == 1);
        REQUIRE(h_pairCount == 0);

    }

    SECTION("One Pair")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 6, 9, 12);

        cudaDeviceSynchronize();

        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_maxRankCount == 2);
        REQUIRE(h_pairCount == 1);
    }

    SECTION("Two Pair")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 3, 3, 4);

        cudaDeviceSynchronize();

        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_maxRankCount == 2);
        REQUIRE(h_pairCount == 2);
    }

    SECTION("3 of a kind")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 3, 3, 3, 4);

        cudaDeviceSynchronize();

        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_maxRankCount == 3);
        REQUIRE(h_pairCount == 0);
    }

    SECTION("Full House")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 3, 3, 3);

        cudaDeviceSynchronize();

        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_maxRankCount == 3);
        REQUIRE(h_pairCount == 1);
    }

    SECTION("4 of a kind")
    {
        testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 3, 3, 3, 3);

        cudaDeviceSynchronize();

        int h_maxRankCount, h_pairCount;
        cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_maxRankCount == 4);
        REQUIRE(h_pairCount == 0);
    }
    // Clean up allocated memory
    cudaFree(d_maxRankCount);
    cudaFree(d_pairCount);
}

TEST_CASE("EvaluatorHelper Test - doFlushAndStaight", "[cuda][kernel][EvaluatorHelper]") 
{
    // Host arrays
    uint8_t ranks[5] = {};
    uint8_t suits[5] = {};

    // device arrays
    uint8_t *d_ranks, *d_suits;
    cudaMalloc(&d_ranks, 5 * sizeof(uint8_t));
    cudaMalloc(&d_suits, 5 * sizeof(uint8_t));

    bool* d_isFlush;
    bool* d_isStraight;
    cudaMalloc(&d_isFlush, sizeof(bool));
    cudaMalloc(&d_isStraight, sizeof(bool));

    // Copy host arrays to device
    cudaMemcpy(d_ranks, ranks, 5 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suits, suits, 5 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    SECTION("straight")
    {
        ranks[0] = 0;
        ranks[1] = 1;
        ranks[2] = 2;
        ranks[3] = 3;
        ranks[4] = 4;

        suits[0] = 0;
        suits[1] = 1;
        suits[2] = 2;
        suits[3] = 3;
        suits[4] = 0;
        testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

        // Synchronize and check for errors
        cudaError_t cudaStatus = cudaDeviceSynchronize();
        REQUIRE(cudaStatus == cudaSuccess);
    
        // Copy results back to HOST
        bool h_isFlush, h_isStraight;
        cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);
    
        CHECK_FALSE(h_isFlush);
        CHECK(h_isStraight);
    }

    SECTION("flush")
    {
        ranks[0] = 0;
        ranks[1] = 1;
        ranks[2] = 2;
        ranks[3] = 3;
        ranks[4] = 8;

        suits[0] = 1;
        suits[1] = 1;
        suits[2] = 1;
        suits[3] = 1;
        suits[4] = 1;

        testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

        // Synchronize and check for errors
        cudaError_t cudaStatus = cudaDeviceSynchronize();
        REQUIRE(cudaStatus == cudaSuccess);

        // Copy results back to HOST
        bool h_isFlush, h_isStraight;
        cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);

        CHECK(h_isFlush);
        CHECK_FALSE(h_isStraight);
    }

    SECTION("Staight Flush")
    {
        ranks[0] = 0;
        ranks[1] = 1;
        ranks[2] = 2;
        ranks[3] = 3;
        ranks[4] = 4;

        suits[0] = 1;
        suits[1] = 1;
        suits[2] = 1;
        suits[3] = 1;
        suits[4] = 1;

        testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

        // Synchronize and check for errors
        cudaError_t cudaStatus = cudaDeviceSynchronize();
        REQUIRE(cudaStatus == cudaSuccess);

        // Copy results back to HOST
        bool h_isFlush, h_isStraight;
        cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);

        CHECK(h_isFlush);
        CHECK(h_isStraight);
    }
    // Cleanup
    cudaFree(d_ranks);
    cudaFree(d_suits);
    cudaFree(d_isFlush);
    cudaFree(d_isStraight);
}

TEST_CASE("EvaluatorHand Tests - operator()", "[EvaluatorHand]") 
{
    int* d_resultScore;
    cudaMalloc(&d_resultScore, sizeof(int));

    SECTION("Royal Flush")
    {
        testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 8, 9, 10, 11, 12);

        cudaDeviceSynchronize();

        int h_resultScore;
        cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_resultScore == 9);
    }
    SECTION("High Card")
    {
        testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 1, 3, 10, 11, 25);

        cudaDeviceSynchronize();
    
        int h_resultScore;
        cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);
    
        REQUIRE(h_resultScore == 0);
    }
    SECTION("Full House")
    {
        testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 1, 14, 15, 28, 41);

        cudaDeviceSynchronize();

        int h_resultScore;
        cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);

        REQUIRE(h_resultScore == 6);
    }

    cudaFree(d_resultScore);
}


TEST_CASE("EvaluatorHand Tests - evaluateAllHands", "[EvaluatorHand]") 
{
    int N = 2;
    thrust::device_vector<Hand> d_hands(N);
    thrust::device_vector<int> d_results(N);

    Hand hand1 = thrust::make_tuple(3, 4, 5, 6, 7);
    Hand hand2 = thrust::make_tuple(8, 9, 10, 11, 12);
    d_hands[0] = hand1;
    d_hands[1] = hand2;

    // Launch kernel with raw pointers
    testEvaluateAllHandsKernel<<<1, N>>>(thrust::raw_pointer_cast(d_hands.data()),
                                        thrust::raw_pointer_cast(d_results.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<int> h_results(N);
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    REQUIRE(h_results[0] == 8);
    REQUIRE(h_results[1] == 9);
}

TEST_CASE("PokerHand - Deal Hand", "[PokerHand]") 
{
    int num_hands = 128;
    curandState* d_rng_states;
    cudaMalloc(&d_rng_states, num_hands * sizeof(curandState));
    
    thrust::device_vector<Hand> d_hands(num_hands);

    generateHandsKernel<<<(num_hands + 31) / 32, 32>>>(
        thrust::raw_pointer_cast(d_hands.data()), num_hands, d_rng_states
    );
    cudaDeviceSynchronize();

    thrust::host_vector<Hand> h_hands(num_hands);

    thrust::copy(d_hands.begin(), d_hands.end(), h_hands.begin());

    CHECK(h_hands.size() == num_hands);

    Hand hand = h_hands[0];
    int cards[5];
    
    cards[0] = thrust::get<0>(hand);
    cards[1] = thrust::get<1>(hand);
    cards[2] = thrust::get<2>(hand);
    cards[3] = thrust::get<3>(hand);
    cards[4] = thrust::get<4>(hand);

    CHECK((cards[0] >= 0 && cards[0] <= 51));

    cudaFree(d_rng_states);
}