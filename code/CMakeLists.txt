cmake_minimum_required(VERSION 3.18)

project(derivatives_pricing_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/home/nicolas/Downloads/libtorch")

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)

add_executable(main
    src/main.cpp
    src/utils/random_generator.cpp
)

target_link_libraries(main
    "${TORCH_LIBRARIES}"
)