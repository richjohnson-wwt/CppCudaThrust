
find_package(Thrust)

add_executable(poker main.cpp)

target_link_libraries(poker PRIVATE thrust::thrust)
