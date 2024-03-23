get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")


# RNG
set(PROJ_NAME "cuda_tree_app_rng")
set(SOURCE
    main_rng.cpp
)
add_executable(${PROJ_NAME})
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_DIR}
        ${CUDA_INCLUDE}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        cuda_tree_lib
        cuda_tree_render
)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES "86;90")

# OBJ
set(PROJ_NAME "cuda_tree_app_obj")
set(SOURCE
    main_obj.cpp
)
add_executable(${PROJ_NAME})
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_DIR}
        ${CUDA_INCLUDE}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        cuda_tree_lib
        cuda_tree_render
)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES "86;90")