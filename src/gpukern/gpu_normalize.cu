#include "gpu_normalize.h"
#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @internal kernel
__global__ void _gpu_normalize_uniform(float* mx , float* my , float* mz, int N){
  int i = threadindex;
  if(i < N){
    float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
}

void gpu_normalize_uniform(float* m, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  _gpu_normalize_uniform<<<gridSize, blockSize>>>(mx, my, mz, N);
  gpu_sync();

}



///@internal kernel
__global__ void _gpu_normalize_map(float* mx , float* my , float* mz, float* normMap, int N){
  int i = threadindex;
  if(i < N){
    float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]) * normMap[i];
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
}

void gpu_normalize_map(float* m, float* map, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  _gpu_normalize_map<<<gridSize, blockSize>>>(mx, my, mz, map, N);
  gpu_sync();
  
}

#ifdef __cplusplus
}
#endif
