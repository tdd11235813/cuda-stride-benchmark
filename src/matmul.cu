// adapted from https://github.com/Kaixhin/cuda-workshop

#include "app_helper.hpp"
#include "cuda_helper.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cassert>

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

template<typename T>
__global__
void matmul_simple(T * const c,
                   T const * const a,
                   T const * const b,
                   const unsigned int width) {

  for (unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
       row < width;
       row += blockDim.x * gridDim.x) {
    for (unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
         col < width;
         col += blockDim.y * gridDim.y) {
      T result = 0;

      for (unsigned int k = 0; k < width; k++) {
        result += a[row * width + k] * b[k * width + col];
      }
      c[row * width + col] = result;
    }
  }
}

// only squared dim allowed (width*width)
// only blockdim = Tilewidth*Tilewidth allowed
// only dim = x * Tilewidth allowed (multiples of Tilewidth)
template<unsigned int Tilewidth, typename T>
__global__ void matmul_tiled(T * const c,
                             T const * const a,
                             T const * const b,
                             const unsigned int width) {

  // Allocate 2D tiles in shared memory
  __shared__ T s_a[Tilewidth][Tilewidth];
  __shared__ T s_b[Tilewidth][Tilewidth];

  const unsigned int nr_tiles_x = width/Tilewidth;
  
  for (unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
       row < width;
       row += blockDim.x * gridDim.x) {
    for (unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
         col < width;
         col += blockDim.y * gridDim.y) {

      T result = 0;

      // Loop over tiles of input in phases
      for (unsigned int p = 0; p < nr_tiles_x; p++) {
        // Collaboratively load tiles into shared memory
        s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*Tilewidth + threadIdx.x)];
        s_b[threadIdx.y][threadIdx.x] = b[(p*Tilewidth + threadIdx.y)*width + col];

        // Wait until all data is loaded before allowing any threads in the block to continue
        __syncthreads();

        // Dot product between row of s_a and column of s_b
        for (unsigned int ti = 0; ti < Tilewidth; ti++) {
          result += s_a[threadIdx.y][ti] * s_b[ti][threadIdx.x];
        }

        // Wait until all calculations are finished before allowing any threads in the block to continue
        __syncthreads();
      }
      // Write result
      c[row * width + col] = result;
    } // col
  } // row
}

// only squared dim allowed (width*width)
// only blockdim = Tilewidth*Tilewidth allowed
// only dim = x * Tilewidth allowed (multiples of Tilewidth)
template<unsigned int Tilewidth, typename T>
__global__ void matmul_tiled_mono(T * const c,
                                  T const * const a,
                                  T const * const b,
                                  const unsigned int width) {

  // Allocate 2D tiles in shared memory
  __shared__ T s_a[Tilewidth][Tilewidth];
  __shared__ T s_b[Tilewidth][Tilewidth];

  // Calculate row and column index of element
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  T result = 0;

  // Loop over tiles of input in phases
  for (unsigned int p = 0; p < width/Tilewidth; p++) {
    // Collaboratively load tiles into shared memory
    s_a[threadIdx.y][threadIdx.x] = a[row*width + (p*Tilewidth + threadIdx.x)];
    s_b[threadIdx.y][threadIdx.x] = b[(p*Tilewidth + threadIdx.y)*width + col];

    // Wait until all data is loaded before allowing any threads in the block to continue
    __syncthreads();

    // Dot product between row of s_a and column of s_b
    for (unsigned int i = 0; i < Tilewidth; i++) {
      result += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
    }

    // Wait until all calculations are finished before allowing any threads in the block to continue
    __syncthreads();
  }

  // Write result
  c[row * width + col] = result;
}

template<int Mode, // 0 = simple, 1 = tiled
         typename T,
         std::uint32_t TRuns,
         std::uint32_t TBlocksizeX,
         std::uint32_t TTilewidthX = TBlocksizeX
         >
void matmul(size_t nx, int dev) {

  bool result_correct = true;
  const std::uint32_t result_correct_nx_threshold = 1024;

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

  CHECK_CUDA( cudaMemcpy( d_matrix_a, h_matrix_a, n_bytes, cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy( d_matrix_b, h_matrix_b, n_bytes, cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy( d_matrix_c, h_matrix_c, n_bytes, cudaMemcpyHostToDevice) );

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev);
  int sqrtNumSMs = std::ceil(std::sqrt(static_cast<double>(numSMs)));

  // TBlocksize would be TBlocksizeX^2.
  dim3 blocksize(TBlocksizeX, TBlocksizeX);
  dim3 blocks_n((nx-1)/TBlocksizeX+1, (nx-1)/TBlocksizeX+1);
  dim3 blocks_i(sqrtNumSMs, sqrtNumSMs);
  assert(blocksize.x == TTilewidthX);
  assert(blocksize.y == TTilewidthX);

  std::uint32_t i=0;
  // -- GRIDSIZE LOOP --
  do{
    // starting with 2*numSMs blocks per grid
    blocks_i.x = std::floor(std::sqrt(2.0)*blocks_i.x+0.5);
    blocks_i.y = std::floor(std::sqrt(2.0)*blocks_i.y+0.5);
    if(blocks_i.x>blocks_n.x)
      blocks_i.x = blocks_n.x;
    if(blocks_i.y>blocks_n.y)
      blocks_i.y = blocks_n.y;
    std::cout << " "
              << std::setw(3) << i++
              << ", " << dev
              << ", " << prop.name
              << ", " << prop.major << '.' << prop.minor
              << ", " << prop.memoryClockRate/1000 // MHz
              << ", " << prop.clockRate/1000 // MHz
              << ", " << get_num_cores(prop)
              << ", " << 2*(prop.clockRate/1000 * get_num_cores(prop)) / 1000
              << ", " << nx
              << ", " << n
              << ", " << numSMs
              << ", " << blocks_i.x * blocks_i.y
              << ", " << (blocks_i.x * blocks_i.y)/numSMs
              << ", " << blocks_n.x * blocks_n.y
              << ", " << blocks_i.x
              << ", " << blocks_n.x
              << ", " << TBlocksizeX
              << ", " << TBlocksizeX*TBlocksizeX
              << ", " << TTilewidthX
              << ", " << TRuns
      ;

    float milliseconds = 0;
    float min_ms = std::numeric_limits<float>::max();

    // -- REPETITIONS --
    for(std::uint32_t r=0; r<TRuns; ++r) {
      CHECK_CUDA( cudaDeviceSynchronize() );
      CHECK_CUDA(cudaEventRecord(cstart, cstream));

      switch(Mode) {
      case 0:
        matmul_simple<<<blocks_i, blocksize, 0, cstream>>>(
          d_matrix_c,
          d_matrix_a,
          d_matrix_b,
          nx
          );
        break;
      case 1:
        matmul_tiled<TTilewidthX><<<blocks_i, blocksize, 0, cstream>>>(
          d_matrix_c,
          d_matrix_a,
          d_matrix_b,
          nx
          );
        break;
      case 2:
        matmul_tiled_mono<TTilewidthX><<<blocks_n, blocksize, 0, cstream>>>(
          d_matrix_c,
          d_matrix_a,
          d_matrix_b,
          nx
          );
        break;
      }

      CHECK_CUDA( cudaGetLastError() );

      CHECK_CUDA( cudaEventRecord(cend, cstream) );
      CHECK_CUDA( cudaEventSynchronize(cend) );
      CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
      if(milliseconds<min_ms)
        min_ms = milliseconds;
    }

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
      if( !result_correct )
        break;
    }
    std::cout << ", " << min_ms << " ms"
              << ", " << 3*n_bytes/min_ms*1e-6 << " GB/s"
              << ", " << 2.0*n*1e-6*nx/min_ms << " GFLOP/s" // 2*nx^3 ops
              << "\n";

  }while( blocks_i.x < blocks_n.x || blocks_i.y < blocks_n.y);

  delete[] h_matrix_a;
  delete[] h_matrix_b;
  delete[] h_matrix_c;
  CHECK_CUDA(cudaFree(d_matrix_a));
  CHECK_CUDA(cudaFree(d_matrix_b));
  CHECK_CUDA(cudaFree(d_matrix_c));
  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  CHECK_CUDA(cudaStreamDestroy(cstream));

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
  using DATA_TYPE = float;

  const int dev=0;
  int mode = 0; // 0 = simple, 1 = tiled, 2 = tiled-monolithic
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
  if(argc==4) // mode
    mode = atoi(argv[3]);

  if(mode==0)
    print_header_matrix("matmul-simple-grid",nx1,nx2);
  else if(mode==1)
    print_header_matrix("matmul-tiled-grid",nx1,nx2);
  else if(mode==2)
    print_header_matrix("matmul-tiled-mono",nx1,nx2);
  else
    return EXIT_FAILURE;

  try{
    for(unsigned nx=nx1; nx<=nx2; nx<<=1) {
      if(mode==0) {
        matmul<0, DATA_TYPE, REPETITIONS, 16>(nx1, dev);
        matmul<0, DATA_TYPE, REPETITIONS, 32>(nx1, dev);
      } else if(mode==1) {
        matmul<1, DATA_TYPE, REPETITIONS, 16>(nx1, dev);
        matmul<1, DATA_TYPE, REPETITIONS, 32>(nx1, dev);
      } else if (mode==2) {
        matmul<2, DATA_TYPE, REPETITIONS, 16>(nx1, dev);
        matmul<2, DATA_TYPE, REPETITIONS, 32>(nx1, dev);
      }
    }
  }catch(std::runtime_error e){
    std::cerr << "\n" << e.what() << "\n";
    CHECK_CUDA( cudaDeviceReset() );
    return 1;
  }
  CHECK_CUDA( cudaDeviceReset() );
  return 0;
}
