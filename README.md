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

# Debug Config

    conan install . --output-folder=build/Debug --build=missing --settings=build_type=Debug -s arch=armv8
    cd build/Debug # Everything below for debug config is in build/Debug!!!
    
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Debug
    cmake --build . --config Debug

    ./src/poker
    ./test/test_poker
    ctest

    Note - Grok say I may need 'conan install .. --build=missing -s arch=armv8 -s build_type=Debug'

# Tasks

The .vscode/tasks.json file should work for both release and debug.
cmd+shift+b will build the project.

# Debugging Notes

MacOS should use the LLDB debugger. GDB has issues on arm64.
pgrep -l lldb-mi # if memory issues.
kill -9 $PID