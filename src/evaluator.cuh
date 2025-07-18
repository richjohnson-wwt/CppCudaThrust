#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include "poker.cuh"

// class Evaluator
// {
// private:
//     thrust::device_vector<int> m_d_results;
//     const int m_numberOfHands;
//     const int NUM_HAND_TYPES;
// public:
//     __device__ Evaluator(int numberOfHands):
//         m_numberOfHands(numberOfHands), 
//         NUM_HAND_TYPES(10), 
//         m_d_results(numberOfHands) {

//     }

    struct evaluate_hand {
        __device__ void unpackTuple(const Hand& hand, int* cards) const {
            cards[0] = thrust::get<0>(hand);
            cards[1] = thrust::get<1>(hand);
            cards[2] = thrust::get<2>(hand);
            cards[3] = thrust::get<3>(hand);
            cards[4] = thrust::get<4>(hand);
        }
        __device__ void rankAndSuits(int* hand, uint8_t *ranks, uint8_t *suits) const {
            for (int i = 0; i < 5; i++) {
                ranks[i] = hand[i] % 13;
                suits[i] = hand[i] / 13;
            }
        }
        __device__ void bubbleSortHand(uint8_t *hand) const {
            for (int i = 0; i < 5 - 1; i++) {
                for (int j = 0; j < 5 - i - 1; j++) {
                    if (hand[j] > hand[j + 1]) {
                        int temp = hand[j];
                        hand[j] = hand[j + 1];
                        hand[j + 1] = temp;
                    }
                }
            }
        }
        __device__ void doRankAndPairCounts(uint8_t* ranks, int& maxRankCount, int& pairCount) const {
            uint8_t rankCount[13] = {0};
            for (int i = 0; i < 5; i++) {
                rankCount[ranks[i]]++;
            }
        
            for (int i = 0; i < 13; i++) {
                if (rankCount[i] > maxRankCount) {
                    maxRankCount = rankCount[i];
                }
                if (rankCount[i] == 2) {
                    pairCount++;
                }
            }
        }
        __device__ void doFlushAndStaight(uint8_t *ranks, uint8_t* suits, bool& isFlush, bool& isStraight) const {
            isFlush = true;
            for (int i = 1; i < 5; i++) {
                if (suits[i] != suits[0]) {
                    isFlush = false;
                    break;
                }
            }
        
            // Check for straight
            isStraight = (ranks[0] + 4 == ranks[4] && ranks[1] + 3 == ranks[4] && ranks[2] + 2 == ranks[4] && ranks[3] + 1 == ranks[4]) ||
                                (ranks[0] == 0 && ranks[1] == 9 && ranks[2] == 10 && ranks[3] == 11 && ranks[4] == 12);
        
        }

        __device__ int operator()(const Hand& hand) const {
            int cards[5];
            unpackTuple(hand, cards);
        
            uint8_t ranks[5];
            uint8_t suits[5];
        
            rankAndSuits(&cards[0], ranks, suits);
            showHand(ranks, suits);
            bubbleSortHand(ranks);
        
            int maxRankCount = 0;
            int pairCount = 0;
            doRankAndPairCounts(&ranks[0], maxRankCount, pairCount);
        
            bool isFlush = false;
            bool isStraight = false;
            doFlushAndStaight(&ranks[0], &suits[0], isFlush, isStraight);
        
            //showHand(ranks, suits);
        
            if (isFlush && isStraight && ranks[4] == 12) {
                // showHand(ranks, suits);
                return 9; // Royal Flush
            } else if (isFlush && isStraight) {
                // showHand(ranks, suits);
                return 8; // Straight Flush
            } else if (maxRankCount == 4) {
                // showHand(ranks, suits);
                return 7; // Four of a Kind
            } else if (maxRankCount == 3 && pairCount == 1) {
                return 6; // Full House
            } else if (isFlush) {
                return 5; // Flush
            } else if (isStraight) {
                return 4; // Straight
            } else if (maxRankCount == 3) {
                return 3; // Three of a Kind
            } else if (pairCount == 2) {
                return 2; // Two Pair
            } else if (pairCount == 1) {
                return 1; // One Pair
            } else {
                return 0; // High Card
            }
        }

        __device__ void showHand(uint8_t *ranks, uint8_t *suits) const {
            printf("EvalHand:");
            for (int i = 0; i < 5; i++) {
                // printf("x%c%c", ranks[hand[i] % 13], suits[hand[i] / 13]);
                printf("%c%c ", "23456789TJQKA"[ranks[i]], "CDHS"[suits[i]]);
            }
            printf("\n");
        }

        __device__ void tallyResults(thrust::device_vector<int> &handTypeCounts, int* numberOfHands, thrust::device_vector<int> &d_results) {
            // std::vector<long> handTypeCounts(10, 0);
            for (int i = 0; i < 5; i++) {
                if (d_results[i] < 10) {
                    handTypeCounts[d_results[i]]++;
                }
            }

            // for (int i = 0; i < 10; i++) {
            //     std::cout << handTypeNames[i] << ": " << handTypeCounts[i] << " ("
            //         << (double)handTypeCounts[i] / *numberOfHands * 100 << "%)" << std::endl;
            // }
        }

        __device__ void evaluateAllHands2(thrust::device_vector<Hand>& d_hands, thrust::device_vector<int> &d_results) {
            thrust::transform(
                thrust::device,
                d_hands.begin(), 
                d_hands.end(),
                d_results.begin(),
                [] __device__ (const Hand& hand) {
                    return evaluate_hand()(hand);
                });
        }
    };

// };

#endif // EVALUATOR_H