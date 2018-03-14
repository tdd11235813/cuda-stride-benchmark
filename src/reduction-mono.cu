#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>

template<unsigned int TBlocksize, unsigned int TMaxWarpNum, typename T>
__global__
void kernel_reduce(T* x, T* y, unsigned int n)
{
  __shared__ T sdata[TBlocksize];

  unsigned int i = blockIdx.x * TBlocksize + threadIdx.x;

  if(i>=n)
    return;

  // --------
  // Level 1: reduce only one pair
  // --------

  T tsum = x[i]; // avoids using the neutral element of specific reduction operation

  const unsigned int grid_size = gridDim.x*TBlocksize;
  i += grid_size;
  if(i<n) {
    tsum += x[i];
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
  if (threadIdx.x == 0) {
    T block_result = sdata[0];
    if(TMaxWarpNum>0) {
      unsigned warpid,smid;
      asm("mov.u32 %0, %%smid;":"=r"(smid));//get SM id
      asm("mov.u32 %0, %%warpid;":"=r"(warpid));//get warp id within SM

      y[smid * TMaxWarpNum + warpid] += block_result;
    }else{
      atomicAdd(y, block_result);
    }
  }
}

template<typename T, unsigned int TRuns, unsigned int TBlocksize, unsigned int TMaxWarpNum>
void reduce(size_t n, unsigned int dev) {

  static_assert(TMaxWarpNum>0, "TMaxWarpNum>0");

  CHECK_CUDA( cudaSetDevice(dev) );
  CHECK_CUDA( cudaFree(0) ); // force context init (applies clocks before getting props)

  cudaDeviceProp prop;
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));
  cudaStream_t cstream;
  CHECK_CUDA(cudaStreamCreate(&cstream));


  T* h_x = new T[n];
  T* x;
  T* y;
  T* z;
  CHECK_CUDA( cudaMalloc(&x, n*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&y, TMaxWarpNum*numSMs*sizeof(T)) );
  CHECK_CUDA( cudaMalloc(&z, sizeof(T)) );
  for (unsigned int i = 0; i < n; i++) {
    h_x[i] = static_cast<T>(1);
  }
  CHECK_CUDA( cudaMemcpy( x, h_x, n*sizeof(T), cudaMemcpyHostToDevice) );


  dim3 blocks = ( ((n+1)/2)-1)/TBlocksize+1; // ceil(ceil(n/2.0)/TBlocksize)
  dim3 blocks_2 = (TMaxWarpNum*numSMs-1)/TBlocksize+1;

  std::cout << " "
            << std::setw(3) << 0
            << ", " << dev
            << ", " << prop.name
            << ", " << prop.major << '.' << prop.minor
            << ", " << prop.memoryClockRate/1000
            << ", " << prop.clockRate/1000
            << ", " << n
            << ", " << numSMs
            << ", " << blocks.x
            << ", " << blocks.x/numSMs
            << ", " << blocks.x
            << ", " << TBlocksize
            << ", " << TRuns
    ;

  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();

  // -- REPETITIONS --
  for(unsigned int r=0; r<TRuns; ++r) {
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA(cudaEventRecord(cstart, cstream));
    // ok, here we do need neutral element of specific reduce operation
    CHECK_CUDA(cudaMemset(y, 0, TMaxWarpNum*numSMs*sizeof(T)));
    CHECK_CUDA(cudaMemset(z, 0, sizeof(T)));

    kernel_reduce<TBlocksize, TMaxWarpNum><<<blocks, TBlocksize, 0, cstream>>>(x, y, n);
    kernel_reduce<TBlocksize, 0><<<blocks_2, TBlocksize, 0, cstream>>>(y, z, TMaxWarpNum*numSMs);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  T result_gpu;
  CHECK_CUDA( cudaMemcpy( &result_gpu, z, sizeof(T), cudaMemcpyDeviceToHost) );
  // check result
  if( result_gpu != static_cast<T>(n) ) {
    std::cerr << "\n\n" << result_gpu << " != " << n << "\n";
    throw std::runtime_error("RESULT MISMATCH");
  }
  std::cout << ", " << min_ms << " ms"
            << ", " << n*sizeof(T)/min_ms*1e-6 << " GB/s"
            << "\n";

  delete[] h_x;
  CHECK_CUDA(cudaFree(x));
  CHECK_CUDA(cudaFree(y));
  CHECK_CUDA(cudaFree(z));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}

int main(int argc, const char** argv)
{

  static constexpr unsigned int REPETITIONS = 5;
  static constexpr unsigned int MAX_WARPS_PER_SM = 64; // hardware specific
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

  print_header("reduction-mono",n1,n2);

  try{
    for(unsigned n=n1; n<=n2; n<<=1) {
      reduce<DATA_TYPE, REPETITIONS, 64, MAX_WARPS_PER_SM>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 128, MAX_WARPS_PER_SM>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 256, MAX_WARPS_PER_SM>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 512, MAX_WARPS_PER_SM>(n, dev);
      reduce<DATA_TYPE, REPETITIONS, 1024, MAX_WARPS_PER_SM>(n, dev);
    }
  }catch(std::runtime_error e){
    std::cerr << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return 1;
  }

  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
