#include "gpu_zeropad.h"
#include "gpu_conf.h"
#include "gpu_stream.h"
#include <assert.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16

///@todo Many of these functions have unused arguments that I left in for clarity but could be removed.

/// @internal Does padding and unpadding of a 2D matrix, not necessarily by a factor 2
__global__ void _gpu_copy_pad2D(float* source, float* dest,
                                int S1, int S2,
                                int D1, int D2){
  
   int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
   int j = blockIdx.x * BLOCKSIZE + threadIdx.x;

   if (i<S1 && j<S2 && i<D1 && j<D2){ //this check makes it work for padding as well as for unpadding.
                                      //2 separate functions are probably not more efficient due to memory bandwidth limitations
    dest[i*D2 + j] = source[i*S2 + j];
   }
}

/// @internal Does padding of a 2D matrix, not necessarily by a factor 2
void gpu_copy_pad2D_async(float* source, float* dest,
                         int S1, int S2,
                         int D1, int D2){

  assert(S1 <= D1 && S2 <= D2);

  dim3 gridSize(divUp(S2, BLOCKSIZE), divUp(S1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);
  _gpu_copy_pad2D<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);/// @todo STREAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

/// @internal Does unpadding of a 2D matrix, not necessarily by a factor 2
void gpu_copy_unpad2D_async(float* source, float* dest,
                         int S1, int S2,
                         int D1, int D2){

  assert(S1 >= D1 && S2 >= D2);

  dim3 gridSize(divUp(D2, BLOCKSIZE), divUp(D1, BLOCKSIZE), 1);
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  check3dconf(gridSize, blockSize);
  _gpu_copy_pad2D<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2); /// @todo STREAM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}


void gpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){
  
  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

  for(int i=0; i<S0; i++){
    gpu_copy_pad2D_async(&source[i*S1*S2], &dest[i*D1*D2], S1, S2, D1, D2); ///@todo inline call to 2D (see also transpose)
  }
  cudaThreadSynchronize();
}


void gpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

  for(int i=0; i<S0; i++){
    gpu_copy_unpad2D_async(&source[i*S1*S2], &dest[i*D1*D2], S1, S2, D1, D2); ///@todo inline call to 2D (see also transpose)
  }
  cudaThreadSynchronize();
}


#ifdef __cplusplus
}
#endif
