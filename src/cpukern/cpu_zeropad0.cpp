#include "cpu_zeropad.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @internal Does padding and unpadding, not necessarily by a factor 2
// __global__ void _gpuconv2_copy_pad(float* source, float* dest,
//                                    int S1, int S2,                  ///< source sizes Y and Z
//                                    int D1, int D2                   ///< destination size Y and Z
//                                    ){
//   int i = blockIdx.x;
//   int j = blockIdx.y;
//   int k = threadIdx.x;
// 
//   dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
// }


void cpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

//   dim3 gridSize(S0, S1, 1); ///@todo generalize!
//   dim3 blockSize(S2, 1, 1);
//   cpu_checkconf(gridSize, blockSize);
// 
//   _gpuconv2_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, D1, D2);
//   cudaThreadSynchronize();

  ///@todo make more efficient
  for(int i=0; i<S0; i++){
    for(int j=0; j<S1; j++){
      for(int k=0; k<S2; k++){
        dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
      }
    }
  }
}


void cpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

//   dim3 gridSize(D0, D1, 1); ///@todo generalize!
//   dim3 blockSize(D2, 1, 1);
//   cpu_checkconf(gridSize, blockSize);

  for(int i=0; i<D0; i++){
    for(int j=0; j<D1; j++){
      for(int k=0; k<D2; k++){
        dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
      }
    }
  }

}



#ifdef __cplusplus
}
#endif
