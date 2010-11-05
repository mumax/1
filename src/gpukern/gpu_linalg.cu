#include "gpu_linalg.h"
#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


///@internal
__global__ void _gpu_add(float* a, float* b, int N){
  int i = threadindex;
  if(i < N){
    a[i] += b[i];
  }
}

void gpu_add(float* a, float* b, int N){
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_add<<<gridSize, blockSize>>>(a, b, N);
  gpu_sync();
}


///@internal
__global__ void _gpu_madd(float* a, float cnst, float* b, int N){
  int i = threadindex;
  if(i < N){
    a[i] += cnst * b[i];
  }
}

void gpu_madd(float* a, float cnst, float* b, int N){
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_madd<<<gridSize, blockSize>>>(a, cnst, b, N);
  gpu_sync();
}


///@internal
__global__ void _gpu_madd2(float* a, float* b, float* c, int N){
  int i = threadindex;
  if(i < N){
    a[i] += b[i] * c[i];
  }
}

void gpu_madd2(float* a, float* b, float* c, int N){
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_madd2<<<gridSize, blockSize>>>(a, b, c, N);
  gpu_sync();
}



///@internal
__global__ void _gpu_add_constant(float* a, float cnst, int N){
  int i = threadindex;
  if(i < N){
    a[i] += cnst;
  }
}

void gpu_add_constant(float* a, float cnst, int N){
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_add_constant<<<gridSize, blockSize>>>(a, cnst, N);
  gpu_sync();
}


///@internal kernel
__global__ void _gpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){
  int i = threadindex;
  if(i < N){
    a[i] = weightA * a[i] + weightB * b[i];
  }
}

void gpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){ 
  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_linear_combination<<<gridSize, blockSize>>>(a, b, weightA, weightB, N);
  gpu_sync();
}

#ifdef __cplusplus
}
#endif
