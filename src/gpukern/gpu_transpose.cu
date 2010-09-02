#include "gpu_transpose.h"
#include "gpu_conf.h"
#include <assert.h>
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


/// The size of matrix blocks to be loaded into shared memory.
/// @todo: optimize this for Fermi, play with non-square blocks.
#define BLOCKSIZE 16


__global__ void _gpu_transpose_complex(float *input, float *output, int N1, int N2)
{

  int N3 = 2;
  
  __shared__ float block[BLOCKSIZE][BLOCKSIZE][2];

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;

  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;
  
  {
    // "major" indices inside the entire matrix
    int I = BI * BLOCKSIZE + i;
    int J = BJ * BLOCKSIZE + j;
//     int K = k;
    
    if((I < N1) && (J < N2)){
      block[j][i][k] = input[J*N1*N3 + I*N3 + k];
    }
  }
  __syncthreads();

  {
    // Major indices with transposed blocks but not transposed minor indices
    int It = BJ * BLOCKSIZE + i;
    int Jt = BI * BLOCKSIZE + j;

    if((It < N2) && (Jt < N1)){
      output[Jt*N2*N3 + It*N3 + k] = block[i][j][k];
    }
  }
}



void gpu_transpose_complex(float *input, float *output, int N1, int N2){
    N2 /= 2;
//     N3 = 2;
    dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 2);
    _gpu_transpose_complex<<<gridsize, blocksize>>>(input, output, N2, N1);
}

/*
 * Replacing BLOCKSIZE+1 by BLOCKSIZE (generating bank conflicts) barely makes it slower.
 */

/*
 * Transposing a block matrix:
 * 1) Transpose the elements inside each block "internally"
 * 2) Transpose the blocks inside the matrix.
 */
__global__ void _gpu_transpose(float *input, float *output, int N1, int N2)
{
  // With this peculiar size there are no shared memory bank conflicts.
  // See NVIDIA's CUDA examples: "efficient matrix transpose".
  // However, this barely seems to affect performance:
  // removing the "+1" makes it only 5% slower, so no need to worry if
  // something HAS to be implemented with memory bank conflicts.
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


///@internal For debugging only
__global__ void _gpu_transpose_slow(float *input, float *output, int N1, int N2)
{

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;

  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;

  // "major" indices inside the entire matrix
  int I = BI * BLOCKSIZE + i;
  int J = BJ * BLOCKSIZE + j;

  // Major indices with transposed blocks but not transposed minor indices
  int It = BJ * BLOCKSIZE + i;
  int Jt = BI * BLOCKSIZE + j;

  if((I < N1) && (J < N2)){
    output[I * N2 + J] = input[J * N1 + I];
  }

}


void gpu_transpose(float *input, float *output, int N1, int N2){
    dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_transpose<<<gridsize, blocksize>>>(input, output, N2, N1);
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
