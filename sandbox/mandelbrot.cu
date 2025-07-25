#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

// Image parameters
const int WIDTH = 3840;
const int HEIGHT = 2160;
const int MAX_ITER = 10000;

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel
__global__ void mandelbrotKernel(unsigned char* image, int width, int height, int maxIter, bool zoom) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float scale;
    float centerX;
    float centerY;
    if (zoom) {
        scale = 0.005f;  // Smaller = more zoom
        centerX = -0.743643887037151;  // Interesting detail
        centerY = 0.13182590420533;
    }
    else {
        // Show full Mandelbrot set
        scale = 4.0f;      // Much larger scale
        centerX = -0.5f;   // Center of Mandelbrot set  
        centerY = 0.0f;
    }

    float fx = centerX + ((float)x / width - 0.5f) * scale * width / height;
    float fy = centerY + ((float)y / height - 0.5f) * scale;

    float zx = 0.0f, zy = 0.0f;
    int iter = 0;
    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float xtemp = zx * zx - zy * zy + fx;
        zy = 2.0f * zx * zy + fy;
        zx = xtemp;
        iter++;
    }

    int offset = (y * width + x) * 3;
    int color = (int)(255.0f * iter / maxIter);

    image[offset + 0] = color;        // R
    image[offset + 1] = color;        // G
    image[offset + 2] = color;        // B
}

int main() {
    size_t imageSize = WIDTH * HEIGHT * 3;
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    unsigned char* d_image;

    CUDA_CHECK(cudaMalloc((void**)&d_image, imageSize));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    int total_blocks = gridSize.x * gridSize.y * gridSize.z;
    int threads_per_block = blockSize.x * blockSize.y * blockSize.z;
    std::cout << "Using " << total_blocks << " blocks with " << threads_per_block << " threads each\n";

    bool zoom = true;
    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT, MAX_ITER, zoom);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost));

    // Save image using stb_image_write
    if (!stbi_write_png("mandelbrot.png", WIDTH, HEIGHT, 3, h_image, WIDTH * 3)) {
        fprintf(stderr, "Failed to write image.\n");
        return 1;
    }

    printf("Image written to mandelbrot.png\n");

    // Cleanup
    cudaFree(d_image);
    free(h_image);

    return 0;
}