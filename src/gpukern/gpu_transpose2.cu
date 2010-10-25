#include "../macros.h"
#include "gpu_transpose2.h"
#include "gpu_conf.h"
#include <assert.h>
#include "timer.h"
#include "gpu_mem.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  float real;
  float imag;
}complex;

/// The size of matrix blocks to be loaded into shared memory.
/// @todo: optimize this for Fermi, play with non-square blocks.
#define BLOCKSIZE 16


__global__ void _gpu_transpose_complex_offset(complex* input, complex* output, int N1, int N2, int offset_in, int offset_out){
  
  __shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;

  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;


  // "major" indices inside the entire matrix
  int I = BI * BLOCKSIZE + i;
  int J = BJ * BLOCKSIZE + j;

  if((I < N1) && (J < N2)){
    block[j][i] = input[J * (N1 + offset_in) + I];
  }
  __syncthreads();


  // Major indices with transposed blocks but not transposed minor indices
  int It = BJ * BLOCKSIZE + i;
  int Jt = BI * BLOCKSIZE + j;

  if((It < N2) && (Jt < N1)){
    output[Jt * (N2+offset_out) + It] = block[i][j];
  }

  return;
}

void gpu_transpose_complex_offset(float *input, float *output, int N1, int N2, int offset_in, int offset_out){
  N2 /= 2;

  dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  _gpu_transpose_complex_offset<<<gridsize, blocksize>>>((complex*)input, (complex*)output, N2, N1, offset_in, offset_out);
}






__global__ void _gpu_transpose_complex_in_plane_fw(complex* input, complex* output, int N1, int N2, int offset1, int offset2){
  
  __shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;

  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;


  // "major" indices inside the entire matrix
  int I = BI * BLOCKSIZE + i;
  int J = BJ * BLOCKSIZE + j;

  if((I < N1) && (J < N2)){
    int ind = J * N1 + I;
    if ( (ind % (2*N2))>=N2 )
      ind += offset2;
    block[j][i] = input[ind];
    input[ind].real = 0.0f;
    input[ind].imag = 0.0f;
  }


__syncthreads();

  // Major indices with transposed blocks but not transposed minor indices
  int It = BJ * BLOCKSIZE + i;
  int Jt = BI * BLOCKSIZE + j;

  if((It < N2) && (Jt < N1)){
    output[Jt * (N2+offset1) + It] = block[i][j];
  }
  
  return;
}


void gpu_transpose_complex_in_plane_fw(float *data, int N1, int N2){

  N2 /= 2;
  int N1x2 = 2*N1;

  timer_start("fw_yz_transpose_copy");
  for(int k=N2+1; k<2*N2; k=k+2){
    int ind1 = k*N1x2;
    int ind2 = (k - N2)*N1x2;
//     printf("%d %d \n", ind1, ind2 );
    memcpy_on_gpu_async(data + ind1, data + ind2, N1x2);
  }
  gpu_sync();
  timer_stop("fw_yz_transpose_copy");

//  _gpu_yz_transpose_copy<<<gridsize, blocksize>>> (data, N2, N1x2);

  timer_start("fw_yz_transpose_transp");
  dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
 _gpu_transpose_complex_in_plane_fw<<<gridsize, blocksize>>>((complex*) (data + N1x2*N2), (complex*)data, N2, N1, N1, -N1*N2);
  timer_stop("fw_yz_transpose_transp");

  return;
}



__global__ void _gpu_transpose_complex_in_plane_inv(complex* input, complex* output, int N1, int N2, int offset_in, int offset2){
  
__shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  // index of the block inside the blockmatrix
  int BI = blockIdx.x;
  int BJ = blockIdx.y;

  // "minor" indices inside the tile
  int i = threadIdx.x;
  int j = threadIdx.y;

  
  // "major" indices inside the entire matrix
  int I = BI * BLOCKSIZE + i;
  int J = BJ * BLOCKSIZE + j;

  if((I < N1) && (J < N2)){
    int ind = J * (N1+offset_in) + I;
    if ( J>(N2/2) )
      ind += offset2;
    block[j][i] = input[ind];
  }
  __syncthreads();


  // Major indices with transposed blocks but not transposed minor indices
  int It = BJ * BLOCKSIZE + i;
  int Jt = BI * BLOCKSIZE + j;

  if((It < N2) && (Jt < N1)){
    output[Jt * N2 + It] = block[i][j];
  }

  return;
}

void gpu_transpose_complex_in_plane_inv(float *data, int N1, int N2){


  N2 /= 2;
  int N2x2 = 2*N2;

  for(int k=N1+1; k<2*N1; k=k+2){
    int ind1 = k*N2x2;
    int ind2 = (k - N1)*N2x2;
    memcpy_on_gpu_async(data + ind1, data + ind2, N2x2);
  }
  gpu_sync();

  dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  _gpu_transpose_complex_in_plane_inv<<<gridsize, blocksize>>>((complex*) data, (complex*)(data + N2x2*N1), N2, N1, N2, -N1*N2);

  return;
}









#ifdef __cplusplus
}
#endif
