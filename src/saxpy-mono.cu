#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>

template<typename T>
__global__
void kernel_saxpy(T *x, T *y, int n, T a) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n)
    y[i] = a * x[i] + y[i];
}


template<typename T, int TRuns, int TBlocksize>
void saxpy(size_t n, int dev) {

  CHECK_CUDA( cudaSetDevice(dev) );
  CHECK_CUDA( cudaFree(0) ); // force context init (applies clocks before getting props)

  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  cudaStream_t cstream;
  CHECK_CUDA(cudaStreamCreate(&cstream));


  const T a = static_cast<T>(42);
  T* h_x = new T[n];
  T* h_y = new T[n];
  T* h_z = new T[n];
  T* x;
  T* y;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, n*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = static_cast<T>(1);
    h_y[i] = static_cast<T>(2);
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);

  int blocks_n = (n-1)/TBlocksize+1;

  std::cout << " "
            << std::setw(3) << 0
            << ", " << dev
            << ", " << prop.name
            << ", " << prop.major << '.' << prop.minor
            << ", " << prop.memoryClockRate/1000
            << ", " << prop.clockRate/1000
            << ", " << n
            << ", " << numSMs
            << ", " << blocks_n
            << ", " << blocks_n/numSMs
            << ", " << blocks_n
            << ", " << TBlocksize
            << ", " << TRuns
    ;

  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  // -- REPETITIONS --
  for(int r=0; r<TRuns; ++r) {
    CHECK_CUDA( cudaMemcpy( y, h_y, n*sizeof(T), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA( cudaEventRecord(cstart, cstream));

    kernel_saxpy<<<blocks_n, TBlocksize, 0, cstream>>>(x, y, n, a);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  CHECK_CUDA( cudaMemcpy( h_z, y, n*sizeof(T), cudaMemcpyDeviceToHost) );
  // check result
  for(int k=0; k<n; ++k) {
    if( h_z[k] != 1*a+2 ) {
      std::cerr << "\n\n y[" << k << "] = " << h_z[k] << "\n";
      throw std::runtime_error("RESULT MISMATCH");
    }
  }
  std::cout << ", " << min_ms << " ms"
            << ", " << 3*n*sizeof(T)/min_ms*1e-6 << " GB/s"
            << "\n";

  delete[] h_x;
  delete[] h_y;
  delete[] h_z;
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}

int main(int argc, const char** argv)
{

  static constexpr int REPETITIONS = 3;
  using DATA_TYPE = int;

  const int dev=0;
  unsigned int n1 = 0;
  unsigned int n2 = 0;
  if(argc>=2)
    n1 = atoi(argv[1]);
  if(n1<2)
    n1 = 1<<28;
  if(argc==3) // range
    n2 = atoi(argv[2]);
  if(n2<n1)
    n2 = n1;

  print_header("saxpy-mono",n1,n2);

  for(unsigned n=n1; n<=n2; n<<=1) {
    saxpy<DATA_TYPE, REPETITIONS, 64>(n, dev);
    saxpy<DATA_TYPE, REPETITIONS, 128>(n, dev);
    saxpy<DATA_TYPE, REPETITIONS, 256>(n, dev);
    saxpy<DATA_TYPE, REPETITIONS, 512>(n, dev);
    saxpy<DATA_TYPE, REPETITIONS, 1024>(n, dev);
  }

  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
