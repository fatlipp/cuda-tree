set(PROJ_NAME "cuda_tree_render")

add_library(${PROJ_NAME})

target_sources(${PROJ_NAME}
    PRIVATE
        base/Render.cpp
        base/RenderCamera.cpp
        tools/OpenGlHelper.cpp
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${PROJECT_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        GL
        GLU
        glfw
        GLEW
)