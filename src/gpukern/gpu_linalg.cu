/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

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


///@internal kernel
__global__ void _gpu_linear_combination_many(float* result, float** vectors, float* weights, int NVectors, int NElem){
  int i = threadindex;
  float result_i = result[i];
  
  if(i < NElem){  
    for(int j=0; j<NVectors; j++){
      result_i += weights[j] * vectors[j][i];
    }
    result[i] = result_i;
  }
}


void gpu_linear_combination_many(float* result, float** vectors, float* weights, int NVectors, int NElem){
  dim3 gridSize, blockSize;
  make1dconf(NElem, &gridSize, &blockSize);
  _gpu_linear_combination_many<<<gridSize, blockSize>>>(result, vectors, weights, NVectors, NElem);
  gpu_sync();
}


///@internal kernel
__global__ void _gpu_scale_dot_product(float* result, float* vector1, float* vector2, float a, int N){
  int i = threadindex;
 
  if(i < N)
    result[i] = a*(vector1[    i] * vector2[    i] + 
                   vector1[  N+i] * vector2[  N+i] + 
                   vector1[2*N+i] * vector2[2*N+i]   );
}

void gpu_scale_dot_product(float* result, float *vector1, float *vector2, float a, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  _gpu_scale_dot_product<<<gridSize, blockSize>>>(result, vector1, vector2, a, N);
  gpu_sync();
}


#ifdef __cplusplus
}
#endif
