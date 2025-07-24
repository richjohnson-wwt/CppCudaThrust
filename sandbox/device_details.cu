#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found.\n";
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << "\n";

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "\nDevice " << device << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock >> 10) << " KB\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Max grid size: (" 
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        std::cout << "  Max block dimensions: (" 
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")\n";
    }

    return 0;
}
