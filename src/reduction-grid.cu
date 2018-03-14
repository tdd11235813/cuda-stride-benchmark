#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>

template<unsigned int TBlocksize, typename T>
__global__
void kernel_reduce(T* x, T* y, unsigned int n)
{
  __shared__ T sdata[TBlocksize];

  unsigned int i = blockIdx.x * TBlocksize + threadIdx.x;

  if(i>=n)
    return;

  T tsum = x[i]; // avoids using the neutral element of specific reduction operation

  const unsigned int grid_size = gridDim.x*TBlocksize;
  i += grid_size;

  // --------
  // Level 1: grid reduce, reading from global memory
  // --------

  // reduce per thread with increased ILP by 4x unrolling sum.
  // the thread of our block reduces its 4 grid-neighbored threads and advances by grid-striding loop
  // (maybe 128bit load improve perf)
  while (i+3*grid_size < n) {
    tsum += x[i] + x[i+grid_size] + x[i+2*grid_size] + x[i+3*grid_size];
    i += 4*grid_size;
  }
  // doing the remaining blocks
  while(i<n) {
    tsum += x[i];
    i += grid_size;
  }

  sdata[threadIdx.x] = tsum;

  __syncthreads();

  // --------
  // Level 2: block + warp reduce, reading from shared memory
  // --------

#pragma unroll
  for(unsigned int bs=TBlocksize,
        bsup=(TBlocksize+1)/2; // ceil(TBlocksize/2.0)
      bs>1;
      bs=bs/2,
        bsup=(bs+1)/2) // ceil(bs/2.0)
  {
    bool cond = threadIdx.x < bsup // only first half of block is working
               && (threadIdx.x+bsup) < TBlocksize // index for second half must be in bounds
               && (blockIdx.x*TBlocksize+threadIdx.x+bsup)<n; // if elem in second half has been initialized before
    if(cond)
    {
      sdata[threadIdx.x] += sdata[threadIdx.x + bsup];
    }
    __syncthreads();
  }

  // store block result to gmem
  if (threadIdx.x == 0)
    y[blockIdx.x] = sdata[0];
}

template<typename T, unsigned int TRuns, unsigned int TBlocksize>
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
  for (unsigned int i = 0; i < n; i++) {
    h_x[i] = static_cast<T>(1);
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);

  unsigned int blocks_i = numSMs;
  unsigned int blocks_n = ( ((n+1)/2)-1)/TBlocksize+1; // ceil(ceil(n/2.0)/TBlocksize)
  unsigned int i=0;
  // -- GRIDSIZE LOOP --
  do{
    blocks_i <<= 1; // starting with 2*numSMs blocks per grid
    if(blocks_i>blocks_n)
      blocks_i = blocks_n;
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
    for(unsigned int r=0; r<TRuns; ++r) {
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

  static constexpr unsigned int REPETITIONS = 5;
  using DATA_TYPE = unsigned;

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

  try{
    for(unsigned n=n1; n<=n2; n<<=1) {
      reduce<DATA_TYPE, REPETITIONS, 64>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 128>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 256>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 512>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 1024>(n, dev);
    }
  }catch(std::runtime_error e){
    std::cerr << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return 1;
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
