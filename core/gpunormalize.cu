#include "gpunormalize.h"

#ifdef __cplusplus
extern "C" {
#endif


void gpu_normalize(param* p, tensor* m){

  int complen = m->len / 3;

  int gridSize = -1, blockSize = -1;
  make1dconf(complen, &gridSize, &blockSize);

  float* mx = &(m->list[0*complen]);
  float* my = &(m->list[1*complen]);
  float* mz = &(m->list[2*complen]);

  timer_start("normalize");
  if(p->msatMap == NULL){
    _gpu_normalize<<<gridSize, blockSize>>>(mx, my, mz);
  }
  else{
    _gpu_normalize_map<<<gridSize, blockSize>>>(mx, my, mz, p->msatMap->list);
  }
  cudaThreadSynchronize();
  timer_stop("normalize");
}


__global__ void _gpu_normalize(float* mx , float* my , float* mz){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}


__global__ void _gpu_normalize_map(float* mx , float* my , float* mz, float* normMap){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]) * normMap[i];
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}



#ifdef __cplusplus
}
#endif