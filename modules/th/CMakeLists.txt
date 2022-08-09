cmake_minimum_required(VERSION 3.7)

project(ThirdLab)
#set(PROJECT_NAME SecondLab)
#set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} src/main.cpp ${CMAKE_SOURCE_DIR}/init.h ${CMAKE_SOURCE_DIR}/init.cpp)

find_package(OpenCL REQUIRED)
find_package(OpenMP)
#target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
if (OpenCL_FOUND AND OpenMP_CXX_FOUND)
    message("Found libraries")
    target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL OpenMP::OpenMP_CXX)
endif()

#set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels $<TARGET_FILE_DIR:${PROJECT_NAME}>/kernels)