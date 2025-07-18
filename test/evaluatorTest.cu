#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/poker.cuh"
#include "../src/evaluator.cuh"

using Hand = thrust::tuple<int, int, int, int, int>;

__global__ void testUnpackTupleKernel(int* resultCards, int a, int b, int c, int d, int e) {
    evaluate_hand testObject; // Create an instance of evaluate_hand
    Hand hand = thrust::make_tuple(a, b, c, d, e);
    testObject.unpackTuple(hand, resultCards); // Call unpackTuple
}

__global__ void testRankAndSuitsKernel(uint8_t* resultRanks, uint8_t* resultSuits, int a, int b, int c, int d, int e) {
    evaluate_hand testObject; // Create an instance of evaluate_hand
    int hand[5] = {a, b, c, d, e};
    testObject.rankAndSuits(&hand[0], resultRanks, resultSuits);
}

__global__ void testBubbleSortHandKernel(uint8_t* resultCards) {
    evaluate_hand testObject; // Create an instance of evaluate_hand
    testObject.bubbleSortHand(resultCards);
    printf("\n\nResult: %d, %d, %d, %d, %d\n\n", resultCards[0], resultCards[1], resultCards[2], resultCards[3], resultCards[4]);
}

__global__ void testRankAndPairCountsKernel(int* rankCount, int* pairCount, uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) {
    evaluate_hand testObject; // Create an instance of evaluate_hand
    uint8_t ranks[5] = {a, b, c, d, e};
    testObject.doRankAndPairCounts(ranks, *rankCount, *pairCount);
}

__global__ void testFlushAndStaightKernel(bool* resultIsFlush, bool* resultIsStraight, uint8_t* ranks, uint8_t* suits) {
    evaluate_hand testObject; // Create an instance of evaluate_hand
    testObject.doFlushAndStaight(&ranks[0], &suits[0], *resultIsFlush, *resultIsStraight);
}

__global__ void testOperatorOverloadKernel(int* resultScore, int a, int b, int c, int d, int e) {
    // evaluate_hand testObject; // Create an instance of evaluate_hand
    Hand hand = thrust::make_tuple(a, b, c, d, e);
    *resultScore = evaluate_hand()(hand);
}

__global__ void testEvaluateAllHandsKernel(Hand* d_hands_ptr, int* d_results_ptr, int count) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < count) {
        d_results_ptr[i] = evaluate_hand{}(d_hands_ptr[i]);
    }
}

TEST_CASE("Evaluator Tests - unpackTuple", "[evaluator]") {
    int host_cards[5] = {0, 0, 0, 0, 0};

    int* device_cards;

    cudaMalloc(&device_cards, sizeof(int) * 5);

    testUnpackTupleKernel<<<1, 1>>>(device_cards, 6, 15, 48, 13, 3);

    cudaDeviceSynchronize();

    cudaMemcpy(host_cards, device_cards, sizeof(int) * 5, cudaMemcpyDeviceToHost);
    cudaFree(device_cards);

    REQUIRE(host_cards[0] == 6);
    REQUIRE(host_cards[1] == 15);
    REQUIRE(host_cards[2] == 48);
    REQUIRE(host_cards[3] == 13);
    REQUIRE(host_cards[4] == 3);
}

TEST_CASE("Evaluator Tests - rankAndSuits", "[evaluator]") {
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

TEST_CASE("Evaluator Test - bubbleSortHand", "[evaluator]") {

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

TEST_CASE("Evaluator Test - doRankAndPairCounts HighCard", "[evaluator]") {
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 6, 2, 9, 0, 3);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 1);
    REQUIRE(h_pairCount == 0);
}

TEST_CASE("Evaluator Test - doRankAndPairCounts One Pair", "[evaluator]") {
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 6, 9, 12);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 2);
    REQUIRE(h_pairCount == 1);
}

TEST_CASE("Evaluator Test - doRankAndPairCounts Two Pair", "[evaluator]") {
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 3, 3, 4);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 2);
    REQUIRE(h_pairCount == 2);
}

TEST_CASE("Evaluator Test - doRankAndPairCounts Trips", "[evaluator]") {

    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 3, 3, 3, 4);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 3);
    REQUIRE(h_pairCount == 0);
}

TEST_CASE("Evaluator Test - doRankAndPairCounts FullHouse", "[evaluator]") {
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 2, 3, 3, 3);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 3);
    REQUIRE(h_pairCount == 1);
}

TEST_CASE("Evaluator Test - doRankAndPairCounts Quad", "[evaluator]") {
    int* d_maxRankCount, *d_pairCount;
    cudaMalloc(&d_maxRankCount, sizeof(int));
    cudaMalloc(&d_pairCount, sizeof(int));

    testRankAndPairCountsKernel<<<1, 1>>>(d_maxRankCount, d_pairCount, 2, 3, 3, 3, 3);

    cudaDeviceSynchronize();

    int h_maxRankCount, h_pairCount;
    cudaMemcpy(&h_maxRankCount, d_maxRankCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pairCount, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_maxRankCount == 4);
    REQUIRE(h_pairCount == 0);
}

TEST_CASE("Evaluator Test - doFlushAndStaight staight", "[evaluator]") {
    // Host arrays
    uint8_t ranks[5] = {0, 1, 2, 3, 4};
    uint8_t suits[5] = {0, 1, 2, 3, 0};

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

    testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

    // Synchronize and check for errors
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    REQUIRE(cudaStatus == cudaSuccess);

    // 4. Copy results back to HOST
    bool h_isFlush, h_isStraight;
    cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);

    CHECK_FALSE(h_isFlush);
    CHECK(h_isStraight);

    // 6. Cleanup
    cudaFree(d_isFlush);
    cudaFree(d_isStraight);
}

TEST_CASE("Evaluator Test - doFlushAndStaight flush", "[evaluator]") {
    uint8_t ranks[5] = {1, 1, 2, 3, 4};
    uint8_t suits[5] = {1, 1, 1, 1, 1};

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

    testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

    // Synchronize and check for errors
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    REQUIRE(cudaStatus == cudaSuccess);

    // 4. Copy results back to HOST
    bool h_isFlush, h_isStraight;
    cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);

    CHECK(h_isFlush);
    CHECK_FALSE(h_isStraight);

    // 6. Cleanup
    cudaFree(d_isFlush);
    cudaFree(d_isStraight);
}

TEST_CASE("Evaluator Test - doFlushAndStaight straight and flush", "[evaluator]") {
    uint8_t ranks[5] = {0, 1, 2, 3, 4};
    uint8_t suits[5] = {1, 1, 1, 1, 1};

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

    testFlushAndStaightKernel<<<1, 1>>>(d_isFlush, d_isStraight, ranks, suits);

    // Synchronize and check for errors
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    REQUIRE(cudaStatus == cudaSuccess);

    // 4. Copy results back to HOST
    bool h_isFlush, h_isStraight;
    cudaMemcpy(&h_isFlush, d_isFlush, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_isStraight, d_isStraight, sizeof(bool), cudaMemcpyDeviceToHost);

    CHECK(h_isFlush);
    CHECK(h_isStraight);

    // 6. Cleanup
    cudaFree(d_isFlush);
    cudaFree(d_isStraight);
}

TEST_CASE("Evaluator Tests - operator() Royal Flush", "[evaluator]") {
    int* d_resultScore;
    cudaMalloc(&d_resultScore, sizeof(int));

    testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 8, 9, 10, 11, 12);

    cudaDeviceSynchronize();

    int h_resultScore;
    cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_resultScore == 9);
}

TEST_CASE("Evaluator Tests - operator() High Card", "[evaluator]") {
    int* d_resultScore;
    cudaMalloc(&d_resultScore, sizeof(int));

    testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 1, 3, 10, 11, 25);

    cudaDeviceSynchronize();

    int h_resultScore;
    cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_resultScore == 0);
}

TEST_CASE("Evaluator Tests - operator() Full House", "[evaluator]") {
    int* d_resultScore;
    cudaMalloc(&d_resultScore, sizeof(int));

    testOperatorOverloadKernel<<<1, 1>>>(d_resultScore, 1, 14, 15, 28, 41);

    cudaDeviceSynchronize();

    int h_resultScore;
    cudaMemcpy(&h_resultScore, d_resultScore, sizeof(int), cudaMemcpyDeviceToHost);

    REQUIRE(h_resultScore == 6);
}

TEST_CASE("Evaluator Tests - evaluateAllHands 2 straight flush hands", "[evaluator]") {
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

    thrust::host_vector<int> h_results(N);  // ✅ Resized host vector
    thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

    REQUIRE(h_results[0] == 8);
    REQUIRE(h_results[1] == 9);
}