

find_package(Catch2 REQUIRED)

add_executable(test_evaluator evaluatorTest.cu)
target_link_libraries(test_evaluator PRIVATE Catch2::Catch2WithMain)
set_property(TARGET test_evaluator PROPERTY CUDA_ARCHITECTURES OFF)

include(CTest)
include(Catch)
catch_discover_tests(test_evaluator)