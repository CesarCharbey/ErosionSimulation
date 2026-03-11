cmake_minimum_required(VERSION 3.10)
project(HydraulicErosion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- FIND PACKAGES ---
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# --- IMGUI SOURCES ---
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# --- EXECUTABLE ---
add_executable(hydraulic_erosion 
    hydraulic_erosion.cpp 
    ${IMGUI_SOURCES}
)

# --- INCLUDES ---
target_include_directories(hydraulic_erosion PRIVATE 
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends 
)

# --- LINKING ---
target_link_libraries(hydraulic_erosion
    OpenGL::GL
    glfw
    GLEW::GLEW
    glm::glm
    ${CMAKE_DL_LIBS}
)

target_compile_options(hydraulic_erosion PRIVATE -Wall -Wextra)

# --- SHADERS COPY ---
add_custom_target(UpdateShaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMENT "Syncing shaders to build folder..."
)