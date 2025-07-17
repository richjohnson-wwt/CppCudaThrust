
find_package(Thrust)
find_package(TBB)

add_executable(poker main.cu)

add_compile_definitions(THRUST_DISABLE_THROW=1)

target_link_libraries(poker PRIVATE thrust::thrust onetbb::onetbb)

set_target_properties(poker PROPERTIES
  BUILD_RPATH "$<TARGET_PROPERTY:onetbb::onetbb,INTERFACE_LINK_DIRECTORIES>"
)