cmake_minimum_required (VERSION 2.8)

find_package(CUDA REQUIRED)

set( CUDA_SEPARABLE_COMPILATION ON)

list(APPEND CUDA_NVCC_FLAGS -std=c++11 -lineinfo --relocatable-device-code true)
#list(APPEND CUDA_NVCC_FLAGS -std=c++11 -lineinfo)
#list(APPEND CUDA_NVCC_FLAGS "-g -G -lineinfo")
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52) # ^coop groups only since P100

if(NOT (CUDA_VERSION_STRING VERSION_LESS 8.0))
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
endif()

if(NOT (CUDA_VERSION_STRING VERSION_LESS 9.0))
  list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=compute_70)
endif()

## cub
find_package(Git)
if(NOT GIT_FOUND)
  message(ERROR "Cannot find git.")
endif()
option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)
set(git_protocol "git")
if(NOT USE_GIT_PROTOCOL)
  set(git_protocol "http")
endif()
include(ExternalProject)
ExternalProject_Add(cub
  GIT_REPOSITORY https://github.com/NVlabs/cub.git
  GIT_TAG master
  SOURCE_DIR ${CMAKE_BINARY_DIR}/cub
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_BINARY_DIR}/cub/cub
    ${CMAKE_CURRENT_LIST_DIR}/../ext/include/cub
  )

#set(SOURCES test_arrays.cu)
# cuda_compile_ptx(
#   cuda_ptx_files
#   ${SOURCES}
#   OPTIONS --source-in-ptx
# )
# add_custom_target(ptx ALL
#   DEPENDS ${cuda_ptx_files}
#   SOURCES ${SOURCES}
# )

cuda_add_executable(reduction-cub reduction-cub.cu)
target_include_directories(reduction-cub PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ext/include)
add_dependencies(reduction-cub cub)
#cuda_add_executable(reduction-thrust reduction-thrust.cu) # cannot measure kernel directly, so skip
cuda_add_executable(reduction-grid reduction-grid.cu)
cuda_add_executable(reduction-mono reduction-mono.cu)
cuda_add_executable(saxpy-grid saxpy-grid.cu)
cuda_add_executable(saxpy-mono saxpy-mono.cu)
cuda_add_executable(matmul matmul.cu)
cuda_add_executable(matmul-cublas matmul_cublas.cpp)
cuda_add_cublas_to_target(matmul-cublas)
