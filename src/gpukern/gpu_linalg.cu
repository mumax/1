#include "gpu_linalg.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


///@internal kernel
__global__ void _gpu_add(float* a, float* b){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  a[i] += b[i];
}

void gpu_add(float* a, float* b, int N){
  int gridSize = -1, blockSize = -1;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_add<<<gridSize, blockSize>>>(a, b);
  cudaThreadSynchronize();
}


///@internal kernel
__global__ void _gpu_add_constant(float* a, float cnst){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  a[i] += cnst;
}

void gpu_add_constant(float* a, float cnst, int N){
  int gridSize = -1, blockSize = -1;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_add_constant<<<gridSize, blockSize>>>(a, cnst);
  cudaThreadSynchronize();
}


///@internal kernel
__global__ void _gpu_linear_combination(float* a, float* b, float weightA, float weightB){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  a[i] = weightA * a[i] + weightB * b[i];
}

void gpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){ 
  int gridSize = -1, blockSize = -1;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_linear_combination<<<gridSize, blockSize>>>(a, b, weightA, weightB);
  cudaThreadSynchronize();
}

#ifdef __cplusplus
}
#endif
