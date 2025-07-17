
find_package(Thrust REQUIRED CONFIG)
find_package(Catch2 REQUIRED)
find_package(TBB)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

add_executable(test_poker pokerTests.cu)
target_link_libraries(test_poker PRIVATE Catch2::Catch2WithMain thrust::thrust onetbb::onetbb)

add_compile_definitions(THRUST_DISABLE_THROW=1)

set_target_properties(test_poker PROPERTIES
  BUILD_RPATH "$<TARGET_PROPERTY:onetbb::onetbb,INTERFACE_LINK_DIRECTORIES>"
)

# add_executable(foo foo.cpp)
# target_link_libraries(foo PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
# catch_discover_tests(test_poker)