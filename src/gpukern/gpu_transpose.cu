#include "gpu_transpose.h"
#include "gpu_conf.h"
#include <assert.h>
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif



#define BLOCK_DIM 16


__global__ void _gpu_transpose(float *idata, float *odata, int width, int height){
  __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
  if((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
  if((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}



void gpu_transpose(float *idata, float *odata, int size_x, int size_y){
    dim3 grid((size_x-1) / BLOCK_DIM + 1, (size_y-1) / BLOCK_DIM + 1, 1);
   // dim3 grid((size_x) / BLOCK_DIM, (size_y) / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    _gpu_transpose<<< grid, threads >>>(idata, odata, size_x, size_y);
}



///@internal kernel
__global__ void _gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
    // N0 <-> N2
    // i  <-> k
    int N3 = 2;

    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

    dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
    dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
}



void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
  timer_start("transposeXZ");
  assert(source != dest);{ // must be out-of-place

  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;  ///@todo: should have new variable here!
  //int N3 = 2;

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeXZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();

  }
  /*  else{
    gpu_transposeXZ_complex_inplace(source, N0, N1, N2*2); ///@todo see above
  }*/
  timer_stop("transposeXZ");
}


__global__ void _gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
    // N1 <-> N2
    // j  <-> k

    int N3 = 2;

        int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

//      int index_dest = i*N2*N1*N3 + k*N1*N3 + j*N3;
//      int index_source = i*N1*N2*N3 + j*N2*N3 + k*N3;


    dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
    dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
/*    dest[index_dest + 0] = source[index_source + 0];
    dest[index_dest + 1] = source[index_source + 1];*/
}

void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
  timer_start("transposeYZ");
  assert(source != dest);{ // must be out-of-place

  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;
  //int N3 = 2;

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeYZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
  }
/*  else{
    gpu_transposeYZ_complex_inplace(source, N0, N1, N2*2); ///@todo see above
  }*/
  timer_stop("transposeYZ");
}






#ifdef __cplusplus
}
#endif
