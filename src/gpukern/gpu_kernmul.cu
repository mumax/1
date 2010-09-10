#include "gpu_kernmul.h"
#include "gpu_conf.h"
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
//   cudaThreadSynchronize();
// }



__global__ void _gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                                                     float* fftKxx, float* fftKyy, float* fftKzz,
                                                     float* fftKyz, float* fftKxz, float* fftKxy){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we some shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kxx = fftKxx[e/2];
  float Kyy = fftKyy[e/2];
  float Kzz = fftKzz[e/2];

  float Kyz = fftKyz[e/2];
  float Kxz = fftKxz[e/2];
  float Kxy = fftKxy[e/2];
  
  fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
  fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

  fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
  fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

  fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
  fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;

}


void gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                                         float* fftKxx, float* fftKyy, float* fftKzz,
                                         float* fftKyz, float* fftKxz, float* fftKxy,
                                         int nRealNumbers){
  
  timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);
  
   int gridSize = -1;
   int blockSize = -1;
   make1dconf(nRealNumbers/2, &gridSize, &blockSize);

  _gpu_kernelmul6<<<gridSize, blockSize>>>(
                                      fftMx,  fftMy,  fftMz, 
                                      fftKxx, fftKyy, fftKzz,
                                      fftKyz, fftKxz, fftKxy);
  cudaThreadSynchronize();
  timer_stop("kernel_mul");
}

#ifdef __cplusplus
}
#endif
