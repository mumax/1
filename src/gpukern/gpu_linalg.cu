#include "gpu_linalg.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


///@internal kernel
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
  cudaThreadSynchronize();
}


///@internal kernel
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
  cudaThreadSynchronize();
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
  cudaThreadSynchronize();
}

#ifdef __cplusplus
}
#endif
