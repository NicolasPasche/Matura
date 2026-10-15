cmake_minimum_required(VERSION 3.18)

project(derivatives_pricing_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep compiled executables inside the build directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


# LibTorch configuration
set(CMAKE_PREFIX_PATH "/home/nicolas/Downloads/libtorch")

find_package(CUDAToolkit REQUIRED)

# Compatibility workaround for the CUDA toolkit and LibTorch.
if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)


# Main application
# Keep this target only if src/main.cpp still exists.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

    add_executable(main
        src/main.cpp
        src/utils/random_generator.cpp
    )

    target_link_libraries(main PRIVATE
        "${TORCH_LIBRARIES}"
    )

endif()


# Analytical pricer tests
add_executable(test_analytical_pricer
    tests/test_analytical_pricer.cpp
    src/pricers/analytical_pricer.cpp
    src/utils/random_generator.cpp
)

target_include_directories(test_analytical_pricer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Merton analytical pricer tests
add_executable(test_merton_analytical_pricer
    tests/test_merton_analytical_pricer.cpp
    src/pricers/merton_analytical_pricer.cpp
    src/pricers/analytical_pricer.cpp
    src/utils/random_generator.cpp
)

target_include_directories(test_merton_analytical_pricer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# ANN pricer test
add_executable(test_ann_pricer
    tests/test_ann_pricer.cpp
    src/utils/random_generator.cpp
)

target_include_directories(test_ann_pricer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(test_ann_pricer PRIVATE
    "${TORCH_LIBRARIES}"
)

