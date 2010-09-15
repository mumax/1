#include "gpu_zeropad.h"
#include "gpu_conf.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16

/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad2D(float* source, float* dest,
                                int S1, int S2,
                                int D1, int D2){
  
   int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
   int j = blockIdx.x * BLOCKSIZE + threadIdx.x;

   if (i<S1 && j < S2){
    dest[i*D2 + j] = source[i*S2 + j];
   }
}


void gpu_copy_pad2D(float* source, float* dest,
                         int S1, int S2,
                         int D1, int D2){

  assert(S1 <= D1 && S2 <= D2);

  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);

  _gpu_copy_pad2D<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();
}



/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad(float* source, float* dest,
                                   int S1, int S2,                  ///< source sizes Y and Z
                                   int D1, int D2                   ///< destination size Y and Z
                                   ){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
}


void gpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){
  
  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

  dim3 gridSize(S0, S1, 1); ///@todo generalize!
  dim3 blockSize(S2, 1, 1);
  check3dconf(gridSize, blockSize);

  _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();
}


void gpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  check3dconf(gridSize, blockSize);

  _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();
}


#ifdef __cplusplus
}
#endif
