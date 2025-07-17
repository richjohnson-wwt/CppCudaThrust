# Intial Setup

    uv venv
    source .venv/bin/activate
    uv pip install conan

# Release Config

    conan install . --build=missing --output-folder=build/Release
    cd build/Release # Everything below for release config is in build/Release!!!
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Release
    cmake --build . 

    ./src/poker   
    ./test/test_poker
    ctest

    export LD_LIBRARY_PATH=~/.conan2/p/b/onetb1501c61605790/p/lib:$LD_LIBRARY_PATH

# Debug Config

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug
    cd build/Debug # Everything below for debug config is in build/Debug!!!
    
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Debug
    cmake --build . --config Debug

    ./src/poker
    ./test/test_poker
    ctest

    Note - Grok say I may need 'conan install .. --build=missing -s arch=armv8 -s build_type=Debug'

    export LD_LIBRARY_PATH=~/.conan2/p/b/onetb99d8bc5c38cdc/p/lib:$LD_LIBRARY_PATH

    nvcc -o poker main.cu

    nvcc -o testPoker pokerTests.cu -I/root/.conan2/p/b/catch2947d50bbdf95/p/include/ -L/root/.conan2/p/b/catch2947d50bbdf95/p/lib -lCatch2Maind -lCatch2d

# Tasks

The .vscode/tasks.json file should work for both release and debug.
cmd+shift+b will build the project.

# Debugging Notes

MacOS should use the LLDB debugger. GDB has issues on arm64.
pgrep -l lldb-mi # if memory issues.
kill -9 $PID