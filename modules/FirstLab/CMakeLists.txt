cmake_minimum_required(VERSION 3.7)

project(FirstLab)
#set(PROJECT_NAME FirstLab)

#set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} src/main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)

#set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels $<TARGET_FILE_DIR:${PROJECT_NAME}>/kernels)