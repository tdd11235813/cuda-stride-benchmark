#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <stdio.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

//#include <cub/cub.cuh>

#define CHECK_CUDA(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

/*only check if kernel start is valid*/
#define CHECK_CUDA_KERNEL(...) __VA_ARGS__;CHECK_CUDA(cudaGetLastError())

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

  /* allocate memory for input data on the host */
  thrust::host_vector<T> h_vec(n);
  thrust::fill(h_vec.begin(), h_vec.end(), 1);
  thrust::device_vector<T> d_vec = h_vec; // host to device copy

  T sum = 0;

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

    sum = thrust::reduce(d_vec.begin(), d_vec.end());

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  // check result
  if( result_gpu != static_cast<T>(n) ) {
    std::cerr << "\n\n" << result_gpu << " != " << n << "\n";
    throw std::runtime_error("RESULT MISMATCH");
  }
  std::cout << ", " << min_ms << " ms"
            << ", " << n*sizeof(T)/min_ms*1e-6 << " GB/s"
            << "\n";

  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

}


// https://nvlabs.github.io/cub/
// git clone https://github.com/NVlabs/cub
void run_cub(int n) {
  cudaEvent_t cstart, cend;
  float milliseconds = 0;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));

  float* d_in;
  float* d_out;
  std::vector<float> h_vec(n);
  std::fill(h_vec.begin(), h_vec.end(), 1);
  CHECK_CUDA(cudaMalloc(&d_in, n*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, h_vec.data(), n*sizeof(float), cudaMemcpyHostToDevice));
  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n); // warmup
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n); // warmup
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n); // warmup
  CHECK_CUDA(cudaEventRecord(cstart,0));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
  CHECK_CUDA(cudaEventRecord(cend,0));
  CHECK_CUDA(cudaEventSynchronize(cend));
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, cstart, cend));

  float sum=0;
  CHECK_CUDA(cudaMemcpy(&sum,d_out,sizeof(float),cudaMemcpyDeviceToHost));
  printf("Addition (GPU cub): %.3f\n", sum);
  printf("Time GPU cub = %f ms\n", milliseconds);
  printf(" bandwidth: %.f GB/s\n", n*sizeof(float)/milliseconds*1e-6);
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_temp_storage));
}

/*
 * The main()-function.
 */
int main (int argc, char **argv)
{
  int dev=0;
  int n = 0;
  if(argc==2)
    n = atoi(argv[1]);
  if(n<2)
    n = 1<<28;
  printf("Running n=%d\n", n);
  try{
    cudaDeviceProp prop;
    CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
    printf("> %s\n", prop.name);
    CHECK_CUDA(cudaSetDevice(dev));
    run_thrust(n);
    run_cub(n);
  }catch(...){
    CHECK_CUDA( cudaDeviceReset() ); // always call this at the end of your CUDA program
    return 1;
  }
  CHECK_CUDA( cudaDeviceReset() ); // always call this at the end of your CUDA program
  return 0;
}
