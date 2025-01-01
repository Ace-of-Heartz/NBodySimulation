cmake_minimum_required(VERSION 3.29)
project(NBody LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE HEADER_FILES
        ${CMAKE_SOURCE_DIR}/src/*.hpp
        ${CMAKE_SOURCE_DIR}/src/*.h
)

file(GLOB_RECURSE SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.c
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

find_package(OpenCL REQUIRED)
message(STATUS "OpenCL included at ${OPENCL_INCLUDE_DIR}")

find_package(GLEW REQUIRED)
message(STATUS "GLEW included at ${GLEW_INCLUDE_DIR}")

find_package(OpenGL REQUIRED)
message(STATUS "OpenGL included at ${OPENGL_INCLUDE_DIR}")

find_package(SDL2 REQUIRED)
message(STATUS "SDL2 included at ${SDL2_INCLUDE_DIR}")

find_package(SDL2_image REQUIRED)

find_package(imgui REQUIRED)
message(STATUS "ImGui included at ${imgui_INCLUDE_DIR}")

find_package(OpenGL REQUIRED)

message(STATUS "All packages found and includes")


target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenCL::OpenCL
        SDL2::SDL2
        GLEW::glew
        SDL2_image::SDL2_image
        GLU

)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/kernels
        ${CMAKE_BINARY_DIR}/kernels
        COMMENT "Copy kernels to result directory")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
        COMMENT "Copy shaders to result directory")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets
        ${CMAKE_BINARY_DIR}/assets
        COMMENT "Copy assets to result directory")

set_target_properties(${PROJECT_NAME} PROPERTIES
        C_STANDARD 20
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=210)