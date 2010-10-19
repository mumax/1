#include "gpu_kernmul.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "assert.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


// __global__ void _gpu_extract_real(float* complex, float* real){
//   int e = ((blockIdx.x * blockDim.x) + threadIdx.x);
//   real[e] = complex[2*e];
// }
// 
// void gpu_extract_real(float* complex, float* real, int NReal){
//   
//   int gridSize = -1, blockSize = -1;
//   make1dconf(NReal, &gridSize, &blockSize);
// 
//   _gpu_extract_real<<<gridSize, blockSize>>>(complex, real);
//   gpu_sync();
// }



__global__ void _gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                                float* fftKxx, float* fftKyy, float* fftKzz,
                                float* fftKyz, float* fftKxz, float* fftKxy, int N){
  int i = threadindex;
  int e = 2 * i;

  // we some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes, cleanup indices
  if(i < N){
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kxx = fftKxx[i];
    float Kyy = fftKyy[i];
    float Kzz = fftKzz[i];

    float Kyz = fftKyz[i];
    float Kxz = fftKxz[i];
    float Kxy = fftKxy[i];

    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
  }
}


void gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers){

  timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  dim3 gridSize, blockSize;
  make1dconf(nRealNumbers/2, &gridSize, &blockSize);

  _gpu_kernelmul6<<<gridSize, blockSize>>>(fftMx,  fftMy,  fftMz,
                                           fftKxx, fftKyy, fftKzz,
                                           fftKyz, fftKxz, fftKxy, nRealNumbers/2);
  gpu_sync();
  timer_stop("kernel_mul");
}


__global__ void _gpu_kernelmul4(float* fftMx,  float* fftMy,  float* fftMz,
                                float* fftKxx, float* fftKyy, float* fftKzz, float* fftKyz, int N){
  int i = threadindex;
  int e = 2 * i;

  // we some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes, cleanup indices
  if(i < N){
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kxx = fftKxx[i];
  float Kyy = fftKyy[i];
  float Kyz = fftKyz[i];
  float Kzz = fftKzz[i];
  
  fftMx[e  ] = reMx * Kxx;
  fftMx[e+1] = imMx * Kxx;
  fftMy[e  ] = reMy * Kyy + reMz * Kyz;
  fftMy[e+1] = imMy * Kyy + imMz * Kyz;
  fftMz[e  ] = reMy * Kyz + reMz * Kzz;
  fftMz[e+1] = imMy * Kyz + imMz * Kzz;
  }
  
  return;
}

void gpu_kernelmul4(float *fftMx, float *fftMy, float *fftMz, float *fftKxx, float *fftKyy, float *fftKzz, float *fftKyz, int nRealNumbers){

  timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  dim3 gridSize, blockSize;
  make1dconf(nRealNumbers/2, &gridSize, &blockSize);

  _gpu_kernelmul4<<<gridSize, blockSize>>>(fftMx, fftMy, fftMz, fftKxx, fftKyy, fftKzz, fftKyz, nRealNumbers/2);
  gpu_sync();
  timer_stop("kernel_mul");
 
  return;
}


__global__ void _gpu_kernelmul3(float* fftMy,  float* fftMz,
                                float* fftKyy, float* fftKzz, float* fftKyz, int N){
  int i = threadindex;
  int e = 2 * i;

  // we some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes, cleanup indices
  if(i < N){

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kyy = fftKyy[i];
  float Kyz = fftKyz[i];
  float Kzz = fftKzz[i];
  
  fftMy[e  ] = reMy * Kyy + reMz * Kyz;
  fftMy[e+1] = imMy * Kyy + imMz * Kyz;
  fftMz[e  ] = reMy * Kyz + reMz * Kzz;
  fftMz[e+1] = imMy * Kyz + imMz * Kzz;
  }
  
  return;
}

void gpu_kernelmul3(float *fftMy, float *fftMz, float *fftKyy, float *fftKzz, float *fftKyz, int nRealNumbers){

  timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  dim3 gridSize, blockSize;
  make1dconf(nRealNumbers/2, &gridSize, &blockSize);

  _gpu_kernelmul3<<<gridSize, blockSize>>>(fftMy, fftMz, fftKyy, fftKzz, fftKyz, nRealNumbers/2);
  gpu_sync();
  timer_stop("kernel_mul");
 
  return;
}

#ifdef __cplusplus
}
#endif
