#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>

template<int TBlocksize, typename T>
__device__
T reduce(int tid, T *x, int n) {

  __shared__ T sdata[TBlocksize];

  int i = blockIdx.x * TBlocksize + tid;

  sdata[tid] = 0;

  // --------
  // Level 1: grid reduce, reading from global memory
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 grid-neighbored threads and advances by grid-striding loop
  while (i+3*gridDim.x*TBlocksize < n) {
    sdata[tid] += x[i] + x[i+gridDim.x*TBlocksize] + x[i+2*gridDim.x*TBlocksize] + x[i+3*gridDim.x*TBlocksize];
    i += 4*gridDim.x*TBlocksize;
  }
  // doing the remaining blocks
  while(i<n) {
    sdata[tid] += x[i];
    i += gridDim.x * TBlocksize;
  }

  __syncthreads();

  // --------
  // Level 2: block + warp reduce, reading from shared memory
  // --------

#pragma unroll
  for(int bs=TBlocksize, bsup=(TBlocksize+1)/2;
      bs>1;
      bs=bs/2, bsup=(bs+1)/2) {
    if(tid < bsup && tid+bsup<TBlocksize) {
      sdata[tid] += sdata[tid + bsup];
    }
    __syncthreads();
  }

  return sdata[0];
}

template<int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, int n)
{
  T block_result = reduce<TBlocksize>(threadIdx.x, x, n);

  // store block result to gmem
  if (threadIdx.x == 0)
    y[blockIdx.x] = block_result;
}

template<typename T, int TRuns, int TBlocksize>
void reduce(size_t n, int dev) {

  CHECK_CUDA( cudaSetDevice(dev) );
  CHECK_CUDA( cudaFree(0) ); // force context init

  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  cudaStream_t cstream;
  CHECK_CUDA(cudaStreamCreate(&cstream));


  T* h_x = new T[n];
  T* x;
  T* y;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  for (int i = 0; i < n; i++) {
    h_x[i] = static_cast<T>(1);
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);

  int blocks_i = numSMs;
  int blocks_n = (n-1)/TBlocksize+1;
  int i=0;
  // -- GRIDSIZE LOOP --
  do{
    blocks_i <<= 1; // starting with 2*numSMs blocks per grid
    std::cout << " "
              << std::setw(3) << i++
              << ", " << dev
              << ", " << prop.name
              << ", " << prop.major << '.' << prop.minor
              << ", " << prop.memoryClockRate/1000
              << ", " << prop.clockRate/1000
              << ", " << n
              << ", " << numSMs
              << ", " << blocks_i
              << ", " << blocks_i/numSMs
              << ", " << blocks_n
              << ", " << TBlocksize
              << ", " << TRuns
      ;
    CHECK_CUDA( cudaMalloc(&y, blocks_i*sizeof(T)) );

    float milliseconds = 0;
    float min_ms = std::numeric_limits<float>::max();

    // -- REPETITIONS --
    for(int r=0; r<TRuns; ++r) {
      CHECK_CUDA( cudaDeviceSynchronize() );
      CHECK_CUDA(cudaEventRecord(cstart, cstream));

      kernel_reduce<TBlocksize><<<blocks_i, TBlocksize, 0, cstream>>>(x, y, n);
      kernel_reduce<TBlocksize><<<1, TBlocksize, 0, cstream>>>(y, y, blocks_i);

      CHECK_CUDA( cudaEventRecord(cend, cstream) );
      CHECK_CUDA( cudaEventSynchronize(cend) );
      CHECK_CUDA( cudaGetLastError() );
      CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
      if(milliseconds<min_ms)
        min_ms = milliseconds;
    }

    T result_gpu;
    CHECK_CUDA( cudaMemcpy( &result_gpu, y, sizeof(T), cudaMemcpyDeviceToHost) );
    // check result
    if( result_gpu != static_cast<T>(n) ) {
      std::cerr << "\n\n" << result_gpu << " != " << n << "\n";
      throw std::runtime_error("RESULT MISMATCH");
    }
    std::cout << ", " << min_ms << " ms"
              << ", " << n*sizeof(T)/min_ms*1e-6 << " GB/s"
              << "\n";

    CHECK_CUDA(cudaFree(y));
  }while( blocks_i < blocks_n );

  delete[] h_x;
  CHECK_CUDA(cudaFree(x));
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

  print_header("reduction-grid",n1,n2);

  for(unsigned n=n1; n<=n2; n<<=1) {
    reduce<DATA_TYPE, REPETITIONS, 64>(n, dev);
    reduce<DATA_TYPE, REPETITIONS, 128>(n, dev);
    reduce<DATA_TYPE, REPETITIONS, 256>(n, dev);
    reduce<DATA_TYPE, REPETITIONS, 512>(n, dev);
    reduce<DATA_TYPE, REPETITIONS, 1024>(n, dev);
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
