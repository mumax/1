#include "gpunormalize.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_normalize(float* mx , float* my , float* mz){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

void gpu_normalize(tensor* m){

  int complen = m->len / 3;
    
  int gridSize = -1, blockSize = -1;
  make1dconf(complen, &gridSize, &blockSize);

  float* mx = &(m->list[0*complen]);
  float* my = &(m->list[1*complen]);
  float* mz = &(m->list[2*complen]);
  
  timer_start("normalize");
  _gpu_normalize<<<gridSize, blockSize>>>(mx, my, mz);
  cudaThreadSynchronize();
  timer_stop("normalize");
}

#ifdef __cplusplus
}
#endif