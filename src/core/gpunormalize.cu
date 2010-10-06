#include "gpunormalize.h"

#ifdef __cplusplus
extern "C" {
#endif


__global__ void _gpu_normalize(float* mx , float* my , float* mz, int N){
  
  int i = threadindex;
  if (i<N){
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
  }
  return;
}


__global__ void _gpu_normalize_map(float* mx , float* my , float* mz, float* normMap, int N){
  
  int i = threadindex;
  
  if (i<N){
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]) * normMap[i];
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
  }
  
  return;
}


void gpu_normalize_uniform(float* m, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

//   timer_start("normalize");
  _gpu_normalize<<<gridSize, blockSize>>>(mx, my, mz, N);
  gpu_sync();
//   timer_stop("normalize");
}

void gpu_normalize_map(float* m, float* map, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

//   timer_start("normalize");
  _gpu_normalize_map<<<gridSize, blockSize>>>(mx, my, mz, map, N);
  gpu_sync();
//   timer_stop("normalize");
  
}


void gpu_normalize(param* p, tensor* m){

  int N = m->len / 3;
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m->list[0*N]);
  float* my = &(m->list[1*N]);
  float* mz = &(m->list[2*N]);

//   timer_start("normalize");
  if(p->msatMap == NULL)
    _gpu_normalize<<<gridSize, blockSize>>>(mx, my, mz, N);
  else
    _gpu_normalize_map<<<gridSize, blockSize>>>(mx, my, mz, p->msatMap->list, N);

  gpu_sync();
//   timer_stop("normalize");
}



#ifdef __cplusplus
}
#endif