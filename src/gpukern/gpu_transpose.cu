/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @author Arne
 */
#include "gpu_transpose.h"
#include "gpu_safe.h"
#include "gpu_conf.h"
#include "gpu_stream.h"
#include <assert.h>
#include "timer.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  float real;
  float imag;
}complex;

/// The size of matrix blocks to be loaded into shared memory.
#define BLOCKSIZE 16

///2D
__global__ void _gpu_transpose_complex_YZ(complex* input, complex* output, int N1, int N2, int N)
{
  __shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  for (int x=0; x<N; x++){
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
        block[j][i] = input[x*N1*N2 + J * N1 + I];
      }
    }
    __syncthreads();

    {
      // Major indices with transposed blocks but not transposed minor indices
      int It = BJ * BLOCKSIZE + i;
      int Jt = BI * BLOCKSIZE + j;

      if((It < N2) && (Jt < N1)){
        output[x*N1*N2 + Jt * N2 + It] = block[i][j];
      }
    }
    __syncthreads();
  }
  
  return;
}

/// 2D transpose
void gpu_transpose_complex_YZ(float *input, float *output, int N1, int N2, int N0){
    N2 /= 2;
    dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_transpose_complex_YZ<<<gridsize, blocksize>>>((complex*)input, (complex*)output, N2, N1, N0);
}


__global__ void _gpu_transpose_complex_XZ(complex* input, complex* output, int N1, int N2, int Ny)
{
  __shared__ complex block[BLOCKSIZE][BLOCKSIZE+1];

  for (int y=0; y<Ny; y++){
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
          block[j][i] = input[J * N1*Ny + y*N1 + I];
        }
      }
      __syncthreads();

      {
        // Major indices with transposed blocks but not transposed minor indices
        int It = BJ * BLOCKSIZE + i;
        int Jt = BI * BLOCKSIZE + j;

        if((It < N2) && (Jt < N1)){
          output[Jt * N2*Ny + y*N2 + It] = block[i][j];
        }
      }
      __syncthreads();
  }
  
  return;
}

void gpu_transpose_complex_XZ(float *input, float *output, int N0, int N1, int N2){
    N2 /= 2;
    dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N0-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N0
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);

    _gpu_transpose_complex_XZ<<<gridsize, blocksize>>>((complex*)input, (complex*)output, N2, N0, N1);

}



///@internal kernel
// __global__ void _gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2, int j){
//     // N0 <-> N2
//     // i  <-> k
//     int N3 = 2;
// 
//     int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
//     int k = blockIdx.x * BLOCKSIZE + threadIdx.x;
// 
//     if(i < N0 && k < N2){
//       dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
//       dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
//     }
// }
// 
// ///@todo this implementation is too slow, especially for "thin" geometries
// void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
// //    timer_start("transposeXZ");
//   assert(source != dest);{ // must be out-of-place
// 
//   // we treat the complex array as a N0 x N1 x N2 x 2 real array
//   // after transposing it becomes N0 x N2 x N1 x 2
//   N2 /= 2;  ///@todo: should have new variable here!
//   //int N3 = 2;
// 
//   dim3 gridSize(divUp(N2, BLOCKSIZE), divUp(N0, BLOCKSIZE), 1);
//   dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
//   check3dconf(gridSize, blockSize);
// 
//  for(int j=0; j<N1; j++){
//     //_gpu_transposeXZ_complex<<<gridSize, blockSize, gpu_getstream()>>>(source, dest, N0, N1, N2, j);
//     _gpu_transposeXZ_complex<<<gridSize, blockSize>>>(source, dest, N0, N1, N2, j); ///@todo STREAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (x2)
//   }
//   gpu_sync();
// 
//   }
// //  timer_stop("transposeXZ");
// }
// 




/// 2D transpose
// void gpu_transpose_complex_async(float *input, float *output, int N1, int N2){
//     N2 /= 2;
//     dim3 gridsize((N2-1) / BLOCKSIZE + 1, (N1-1) / BLOCKSIZE + 1, 1); // integer division rounded UP. Yes it has to be N2, N1
//     dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
//     _gpu_transpose_complex<<<gridsize, blocksize>>>((complex*)input, (complex*)output, N2, N1);  /// ///////////////// @todo STREAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// }

///@todo need to time this on 2.0 hardware
// void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
// //   timer_start("transposeYZ");
//   for(int i=0; i<N0; i++){
//     gpu_transpose_complex_async(&source[i*N1*N2], &dest[i*N1*N2], N1, N2);
//   }
//    gpu_sync();
// //    timer_stop("transposeYZ");
// }


#ifdef __cplusplus
}
#endif
