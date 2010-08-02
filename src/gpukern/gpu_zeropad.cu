#include "gpu_zeropad.h"
#include "gpu_conf.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


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


void gpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){
  
  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

  dim3 gridSize(S0, S1, 1); ///@todo generalize!
  dim3 blockSize(S2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();
  
}


void gpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
  cudaThreadSynchronize();

}


#ifdef __cplusplus
}
#endif
