set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA 12.2 REQUIRED)

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")

set(SOURCE
    # CPP
    tools/BlockTimer.cpp
    tools/cuda/CudaHelper.cpp
    # CUDA
    core/quad_tree/QuadTreeBuilderCuda.cu
    core/octree/OctreeBuilderCuda.cu
    tools/cuda/RandomPointsGenerator.cu
    tools/cuda/RandomPointsGenerator3d.cu
)

set(PROJ_NAME "cuda_tree_lib")

add_library(${PROJ_NAME})

target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_DIR}
        ${CUDA_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)
target_link_libraries(${PROJ_NAME} 
PRIVATE
    ${CUDA_LIBRARIES} 
)

set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES "86;90")