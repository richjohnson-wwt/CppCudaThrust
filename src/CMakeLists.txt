

add_executable(poker main.cu poker.cu evaluator.cu)

set_property(TARGET poker PROPERTY CUDA_ARCHITECTURES OFF)