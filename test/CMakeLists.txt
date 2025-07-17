
find_package(Thrust REQUIRED CONFIG)
find_package(Catch2 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

add_executable(test_poker pokerTests.cpp)
target_link_libraries(test_poker PRIVATE Catch2::Catch2WithMain thrust::thrust)

add_executable(foo foo.cpp)
target_link_libraries(foo PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_poker)