#include "app_helper.hpp"
#include "cuda_helper.cuh"
#include <stdio.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <vector>

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
  T* d_in;
  T* d_out;
  std::vector<T> h_vec(n);
  std::fill(h_vec.begin(), h_vec.end(), 1);
  CHECK_CUDA(cudaMalloc(&d_in, n*sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_in, h_vec.data(), n*sizeof(T), cudaMemcpyHostToDevice));
  // Determine temporary device storage requirements
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  T result_gpu = 0;

  dim3 blocks = 2; //( ((n+1)/2)-1)/TBlocksize+1; // ceil(ceil(n/2.0)/TBlocksize)
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

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);

    CHECK_CUDA( cudaEventRecord(cend, cstream) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;
  }

  // check result
  CHECK_CUDA(cudaMemcpy(&result_gpu,d_out,sizeof(T),cudaMemcpyDeviceToHost));
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
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_temp_storage));

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

  print_header("reduction-cub",n1,n2);

  try{
    for(unsigned n=n1; n<=n2; n<<=1) {
      // cub seems to always run 256 threads per block
      reduce<DATA_TYPE, REPETITIONS, 256 /* just for output */, MAX_WARPS_PER_SM>(n, dev);
    }
  }catch(std::runtime_error e){
    std::cerr << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return 1;
  }

  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
