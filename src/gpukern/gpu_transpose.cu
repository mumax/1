#include "gpu_transpose.h"
#include "gpu_conf.h"
#include <assert.h>
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif



#define BLOCKSIZE 16

/*
 * Transposing a block matrix:
 * 1) Transpose the elements inside each block "internally"
 * 2) Transpose the blocks inside the matrix.
 */
__global__ void _gpu_transpose(float *input, float *output, int N1, int N2)
{
  __shared__ float block[BLOCKSIZE][BLOCKSIZE+1];

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;
  
  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;
  
  {
    // "major" indices inside the entire matrix
    int I = BI * BLOCKSIZE + i;
    int J = BJ * BLOCKSIZE + j;
    
    if((I < N1) && (J < N2)){
      block[j][i] = input[J * N1 + I];
    }
  }
  __syncthreads();
  
  {
    // Major indices with transposed blocks but not transposed minor indices
    int It = BJ * BLOCKSIZE + i;
    int Jt = BI * BLOCKSIZE + j;
    
    if((It < N2) && (Jt < N1)){
      output[Jt * N2 + It] = block[i][j];
    }
  }
}




void gpu_transpose(float *input, float *output, int size_y, int size_x){
    dim3 grid((size_x-1) / BLOCKSIZE + 1, (size_y-1) / BLOCKSIZE + 1, 1); // integer division rounded UP
    dim3 threads(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_transpose<<< grid, threads >>>(input, output, size_x, size_y);
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
