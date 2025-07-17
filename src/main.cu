#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

void foo(int sum) {
    
    std::cout << "The sum from thrust is: " << sum << std::endl;
}

int main() {
    #ifdef NDEBUG
        printf("Release configuration!\n");
    #else
        printf("Debug configuration!\n");
    #endif

    std::cout << "Hello, World!" << std::endl;

    std::vector<int> host_vec{3, 1, 4, 1, 5, 9};
    thrust::device_vector<int> vec(host_vec.begin(), host_vec.end());
    thrust::sort(thrust::device, vec.begin(), vec.end());
    int sum = thrust::reduce(thrust::device, vec.begin(), vec.end(), 0);

    foo(sum);

    return EXIT_SUCCESS;
}