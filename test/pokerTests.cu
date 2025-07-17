#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


int helper() {
    std::vector<int> host_vec{3, 1, 4, 1, 5, 9};
    thrust::device_vector<int> vec(host_vec.begin(), host_vec.end());

    std::cout << "Before sort" << std::endl;
    thrust::sort(thrust::device, vec.begin(), vec.end());
    std::cout << "After sort" << std::endl;
    return thrust::reduce(thrust::device, vec.begin(), vec.end(), 0);
}

TEST_CASE("Thrust sort and reduce on CPU (via OpenMP backend)", "[thrust]") {
    int sum = helper();
    REQUIRE(sum == 23);
    // REQUIRE(vec[0] == 1);
    // REQUIRE(vec[5] == 9);
    // int x = 0;
}