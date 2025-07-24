# About

This is a learning project for CUDA development. Setting up a development environment with unit tests was the main purpose. It uses Conan version 2 package manager and CMake. The unit tests run on the GPU as does the main app.

# Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson

    After conan install, update task.json Catch2 path
    ls ~/.conan2/p/b
    set(CMAKE_CUDA_ARCHITECTURES 89) based on the nvidia-smi (ie RTX 5070 Ti)

# Debug Config

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
    cd build/Debug 
    
    # All commands in build/Debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug
    cmake --build .

    ./src/poker
    ./test/test_poker
    ctest

    Note - Grok say I may need 'conan install .. --build=missing -s arch=armv8 -s build_type=Debug'

    nvcc -o poker main.cu

    nvcc -o testPoker pokerTests.cu -I/root/.conan2/p/b/catch2947d50bbdf95/p/include/ -L/root/.conan2/p/b/catch2947d50bbdf95/p/lib -lCatch2Maind -lCatch2d

# Release Config

    conan install . --build=missing --output-folder=build/Release
    cd build/Release 

    # All commands in build/Release!!!
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Release
    cmake --build . 

    ./src/poker   
    ./test/test_poker
    ctest



# Tasks

The .vscode/tasks.json file should work for both release and debug.
cmd+shift+b will build the project.

# Debugging Notes

MacOS should use the LLDB debugger. GDB has issues on arm64.
pgrep -l lldb-mi # if memory issues.
kill -9 $PID
vast.ai - Cannot debug on GPU without special tools: nsight compute, cuda-gdb TODO

# Git

    git config --global user.email "rich.johnson@wwt.com"
    git config --global user.name "Rich Johnson"

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson

# SCP file

    scp -P 55425 root@116.102.206.157:/workspace/CppCudaThrust/src/mandelbrot.png .

# device info 7/24/25

    Device 0: NVIDIA GeForce RTX 5070 Ti
    Compute capability: 12.0
    Total global memory: 15848 MB
    Multiprocessors: 70
    Clock rate: 2452 MHz
    Shared memory per block: 48 KB
    Registers per block: 65536
    Warp size: 32
    Max threads per block: 1024
    Max threads per SM: 1536
    Max grid size: (2147483647, 65535, 65535)
    Max block dimensions: (1024, 1024, 64)