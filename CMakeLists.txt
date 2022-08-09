cmake_minimum_required(VERSION 3.7)

project(OpenCL)
set(CMAKE_CXX_STANDARD 17)

#find_package(OpenCL REQUIRED)

file(GLOB SUBDIRS "${CMAKE_CURRENT_SOURCE_DIR}/modules/*")

foreach(subdir ${SUBDIRS})
  add_subdirectory(${subdir})
endforeach()