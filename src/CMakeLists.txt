cmake_minimum_required (VERSION 2.8)

find_package(CUDA REQUIRED)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(FATAL_ERROR "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "gcc5.0 or newer required.")
  endif()
endif()

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

cuda_add_executable(reduction-grid reduction-grid.cu)
cuda_add_executable(reduction-mono reduction-mono.cu)
cuda_add_executable(saxpy-grid saxpy-grid.cu)
cuda_add_executable(saxpy-mono saxpy-mono.cu)
