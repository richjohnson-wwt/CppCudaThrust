

find_package(Catch2 REQUIRED)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

add_executable(test_poker pokerTests.cu)
target_link_libraries(test_poker PRIVATE Catch2::Catch2WithMain)
set_property(TARGET test_poker PROPERTY CUDA_ARCHITECTURES OFF)

# add_executable(foo foo.cpp)
# target_link_libraries(foo PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_poker)