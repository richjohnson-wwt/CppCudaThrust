#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>


using Hand = thrust::tuple<int, int, int, int, int>;

class PokerHand 
{
public:
    __device__ void dealHand(Hand* d_hands, int thread_id, int seed1) {
        bool used[52] = {false};
        int cards[5];

        // Very simple PRNG for illustration only
        unsigned int seed = simple_rand(thread_id + seed1);

        for (int i = 0; i < 5; ++i) {
            int card;
            do {
                seed = seed * 1664525 + 1013904223;  // LCG
                card = seed % 52;
            } while (used[card]);
            used[card] = true;
            cards[i] = card;
        }

        d_hands[thread_id] = thrust::make_tuple(cards[0], cards[1], cards[2], cards[3], cards[4]);
    }
private:
    __device__ unsigned int simple_rand(int tid, int iter = 0) {
        unsigned int seed = tid * 9781 + iter * 7919 + 17;
        return seed * 1664525 + 1013904223;
    }

};


class EvaluatorHelper
{
public:
    
    __device__ void unpackTuple(const Hand& hand, int* cards) const 
    {
        cards[0] = thrust::get<0>(hand);
        cards[1] = thrust::get<1>(hand);
        cards[2] = thrust::get<2>(hand);
        cards[3] = thrust::get<3>(hand);
        cards[4] = thrust::get<4>(hand);
    }

    __device__ void rankAndSuits(int* hand, uint8_t *ranks, uint8_t *suits) const 
    {
        for (int i = 0; i < 5; i++) {
            ranks[i] = hand[i] % 13;
            suits[i] = hand[i] / 13;
        }
    }

    __device__ void bubbleSortHand(uint8_t *hand) const 
    {
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

    __device__ void doRankAndPairCounts(uint8_t* ranks, int& maxRankCount, int& pairCount) const 
    {
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

    __device__ void doFlushAndStaight(uint8_t *ranks, uint8_t* suits, bool& isFlush, bool& isStraight) const 
    {
        isFlush = true;
        for (int i = 1; i < 5; i++) {
            if (suits[i] != suits[0]) {
                isFlush = false;
                break;
            }
        }

        // Check for straight
        // Normal straight: consecutive ranks
        isStraight = (ranks[0] + 1 == ranks[1] && ranks[1] + 1 == ranks[2] && 
                    ranks[2] + 1 == ranks[3] && ranks[3] + 1 == ranks[4]);
        
        // Special case: A-2-3-4-5 low straight (A=0, 2=1, 3=2, 4=3, 5=4)
        if (!isStraight && ranks[0] == 0 && ranks[1] == 1 && ranks[2] == 2 && ranks[3] == 3 && ranks[4] == 4) {
            isStraight = true;
        }

    }

};

class EvaluatorHand 
{
private:
    EvaluatorHelper m_helper;
public:
    // Main evaluation function - called from kernels
    __device__ int operator()(const Hand& hand) const 
    {
        int cards[5];
        m_helper.unpackTuple(hand, cards);
    
        uint8_t ranks[5];
        uint8_t suits[5];
    
        m_helper.rankAndSuits(&cards[0], ranks, suits);
        // Remove showHand call to avoid garbled output
        m_helper.bubbleSortHand(ranks);
    
        int maxRankCount = 0;
        int pairCount = 0;
        m_helper.doRankAndPairCounts(&ranks[0], maxRankCount, pairCount);
    
        bool isFlush = false;
        bool isStraight = false;
        m_helper.doFlushAndStaight(&ranks[0], &suits[0], isFlush, isStraight);
    
        if (isFlush && isStraight && ranks[4] == 12) {
            // Royal Flush - don't print here due to thread concurrency
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
};


#endif // EVALUATOR_H
