

add_executable(poker main.cu)

set_property(TARGET poker PROPERTY CUDA_ARCHITECTURES OFF)