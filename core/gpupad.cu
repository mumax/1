#include "gpupad.h"

#ifdef __cplusplus
extern "C" {
#endif

//_____________________________________________________________________________________________ copy/pad

/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpuconv2_copy_pad(float* source, float* dest,
                                   int S1, int S2,                  ///< source sizes Y and Z
                                   int D1, int D2                   ///< destination size Y and Z
                                   ){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
}

void gpu_copy_pad_unsafe(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){
  
  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

  dim3 gridSize(S0, S1, 1); ///@todo generalize!
  dim3 blockSize(S2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();
  
}

void gpu_copy_unpad_unsafe(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();

}

void gpu_copy_pad(tensor* source, tensor* dest){

  assert(source->rank == 3);
  assert(  dest->rank == 3);

  // source must not be larger than dest
  for(int i=0; i<3; i++){
    assert(source->size[i] <= dest->size[i]);
  }

  int S0 = source->size[X];
  int S1 = source->size[Y];
  int S2 = source->size[Z];

  gpu_copy_pad_unsafe(source->list, dest->list, S0, S1, S2, dest->size[0], dest->size[1], dest->size[2]);
}


void gpu_copy_unpad(tensor* source, tensor* dest){

  assert(source->rank == 3);
  assert(  dest->rank == 3);

  // dest must not be larger than source
  for(int i=0; i<3; i++){
    assert(source->size[i] >= dest->size[i]);
  }

  int D0 = dest->size[X];
  int D1 = dest->size[Y];
  int D2 = dest->size[Z];
  
  gpu_copy_unpad_unsafe(source->list, dest->list, source->size[0], source->size[1], source->size[2], D0, D1, D2);
}

#ifdef __cplusplus
}
#endif