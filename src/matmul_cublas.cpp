// adapted from https://github.com/Kaixhin/cuda-workshop

#include "app_helper.hpp"
#include "cuda_helper.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <cublas_v2.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cassert>

// specify error checking for cublas

inline
void check_cuda(cublasStatus_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw_error(static_cast<int>(code),
                "CUBLAS error.", msg, func, file, line);
  }
}

template<typename T>
void matmul_cpu(T *c,
                T const * const a,
                T const * const b,
                const unsigned int width) {

  T result;
  for (unsigned int row = 0; row < width; row++) {
    for (unsigned int col = 0; col < width; col++) {
      result = 0;
      for (unsigned int k = 0; k < width; k++) {
        result += a[row * width + k] * b[k * width + col];
      }
      c[row * width + col] = result;
    }
  }
}

template<typename T,
         std::uint32_t TRuns
         >
void matmul(size_t nx, int dev) {

  CHECK_CUDA( cudaSetDevice(dev) );
  CHECK_CUDA( cudaFree(0) ); // force context init

  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  cudaStream_t cstream;
  CHECK_CUDA(cudaStreamCreate(&cstream));

  std::size_t n = nx*nx; // square matrix
  std::size_t n_bytes = n * sizeof(T);
  T* h_matrix_a = new T[n];
  T* h_matrix_b = new T[n];
  T* h_matrix_c = new T[n];
  T* d_matrix_a = 0;
  T* d_matrix_b = 0;
  T* d_matrix_c = 0;

  // Allocation on device

  CHECK_CUDA( cudaMalloc(&d_matrix_a, n_bytes) );
  CHECK_CUDA( cudaMalloc(&d_matrix_b, n_bytes) );
  CHECK_CUDA( cudaMalloc(&d_matrix_c, n_bytes) );

  // Init data on host

  std::srand(1337);
  for (std::uint32_t i = 0; i < n; i++) {
    h_matrix_a[i] = std::rand()/((RAND_MAX + 1u)/6);  // Note: 1+rand()%6 is biased
    h_matrix_b[i] = std::rand()/((RAND_MAX + 1u)/6);  // Note: 1+rand()%6 is biased
    h_matrix_c[i] = 0;
  }

  // Copy to device

  // CHECK_CUDA( cudaMemcpy( d_matrix_a, h_matrix_a, n_bytes, cudaMemcpyHostToDevice) );
  // CHECK_CUDA( cudaMemcpy( d_matrix_b, h_matrix_b, n_bytes, cudaMemcpyHostToDevice) );
  // CHECK_CUDA( cudaMemcpy( d_matrix_c, h_matrix_c, n_bytes, cudaMemcpyHostToDevice) );
  // or
  CHECK_CUDA( cublasSetMatrix(nx, nx, sizeof(T), h_matrix_a, nx, d_matrix_a, nx) );
  CHECK_CUDA( cublasSetMatrix(nx, nx, sizeof(T), h_matrix_b, nx, d_matrix_b, nx) );
  CHECK_CUDA( cublasSetMatrix(nx, nx, sizeof(T), h_matrix_c, nx, d_matrix_c, nx) );

//  std::uint32_t i=0;
  std::cout << " "
            << std::setw(3) << 0
            << ", " << dev
            << ", " << prop.name
            << ", " << prop.major << '.' << prop.minor
            << ", " << prop.memoryClockRate/1000 // MHz
            << ", " << prop.clockRate/1000 // MHz
            << ", " << get_num_cores(prop)
            << ", " << 2*(prop.clockRate/1000 * get_num_cores(prop)) / 1000 // FMA
            << ", " << nx
            << ", " << n
            << ", " << prop.multiProcessorCount
            << ", " << 2
            << ", " << 2
            << ", " << 2
            << ", " << 2
            << ", " << 2
            << ", " << 16
            << ", " << 256
            << ", " << 0
            << ", " << TRuns
    ;

  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  // BLAS GEMM: C = alpha A * B + beta C
  // cuBLAS works in column-major format
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasHandle_t handle;
  CHECK_CUDA(cublasCreate(&handle));

  // -- REPETITIONS --
  for(std::uint32_t r=0; r<TRuns; ++r) {
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA(cudaEventRecord(cstart, cstream));

    //note cublas is column primary!
    //need to transpose the order
    CHECK_CUDA(cublasSgemm(handle,
                           CUBLAS_OP_N, // cublasOperation_t transa (N - non-transpose operation)
                           CUBLAS_OP_N, // cublasOperation_t transb (N - non-transpose operation)
                           nx, // number of rows of matrix op(A) and C
                           nx, // number of columns of matrix op(B) and C
                           nx, // number of columns of op(A) and rows of op(B)
                           &alpha,
                           d_matrix_b, // op(A)
                           nx,
                           d_matrix_a, // op(B)
                           nx,
                           &beta,
                           d_matrix_c,  // op(C)
                           nx));


    CHECK_CUDA( cudaGetLastError() );

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  bool result_correct = true;
  const std::uint32_t result_correct_nx_threshold = 1024;

  // check result
  if( nx<result_correct_nx_threshold ) {
    CHECK_CUDA( cudaMemcpy( h_matrix_c, d_matrix_c, n_bytes, cudaMemcpyDeviceToHost) );
    T* h_matrix = new T[n];
    matmul_cpu(h_matrix, h_matrix_a, h_matrix_b, nx);

    result_correct = true;
    for(std::uint32_t j=0; j<n; ++j) {
      if( h_matrix[j] != h_matrix_c[j] ) { // TODO: traits for fixed-point and floating-point comparison
        std::cerr << "\n\n" << h_matrix_c[j] << " != " << h_matrix[j] << " [i=" <<j<<"]\n";
        result_correct = false;
        break;
      }
    }
    delete[] h_matrix;
  }
  std::cout << ", " << min_ms << " ms"
            << ", " << 3*n_bytes/min_ms*1e-6 << " GB/s" // A,B,C
            << ", " << 2.0*n*1e-6*nx/min_ms << " GFLOP/s" // 2*nx^3 ops
            << "\n";

  delete[] h_matrix_a;
  delete[] h_matrix_b;
  delete[] h_matrix_c;
  CHECK_CUDA(cudaFree(d_matrix_a));
  CHECK_CUDA(cudaFree(d_matrix_b));
  CHECK_CUDA(cudaFree(d_matrix_c));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));
  CHECK_CUDA(cublasDestroy(handle));

  if(result_correct) {
    if(nx<result_correct_nx_threshold) {
      std::cout << "Results are correct.\n";
    }
  } else {
    throw std::runtime_error("RESULT MISMATCH");
  }
}


int main(int argc, const char** argv)
{

  static constexpr std::uint32_t REPETITIONS = 5;
  using DATA_TYPE = float; // we use sgemm, so only float is allowed here

  const int dev=0;
  std::uint32_t nx1 = 0;
  std::uint32_t nx2 = 0;
  if(argc>=2)
    nx1 = atoi(argv[1]);
  if(nx1<2)
    nx1 = 1<<4;
  if(argc==3) // range
    nx2 = atoi(argv[2]);
  if(nx2<nx1)
    nx2 = nx1;

  print_header_matrix("matmul-cublas",nx1,nx2);

  try{
    for(unsigned nx=nx1; nx<=nx2; nx<<=1) {
      matmul<DATA_TYPE, REPETITIONS>(nx1, dev);
    }
  }catch(std::runtime_error e){
    std::cerr << "\n" << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return 1;
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
