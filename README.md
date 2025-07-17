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

# Release Config

    conan install . --build=missing --output-folder=build/Release
    cd build/Release 

    # All commands in build/Release!!!
    cmake ../.. -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=Release
    cmake --build . 

    ./src/poker   
    ./test/test_poker
    ctest

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

# Tasks

The .vscode/tasks.json file should work for both release and debug.
cmd+shift+b will build the project.

# Debugging Notes

MacOS should use the LLDB debugger. GDB has issues on arm64.
pgrep -l lldb-mi # if memory issues.
kill -9 $PID

# Git

    git config --global user.email "rich.johnson@wwt.com"
    git config --global user.name "Rich Johnson"

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson