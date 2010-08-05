#include "cpu_kernmul.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers){

  //timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  //    int gridSize = -1;
  //    int blockSize = -1;
  //    make1dconf(nRealNumbers/2, &gridSize, &blockSize);

  for(int e=0; e<nRealNumbers; e+=2){
    float reMx = fftMx[e  ];
    float imMx = fftMx[e+1];

    float reMy = fftMy[e  ];
    float imMy = fftMy[e+1];

    float reMz = fftMz[e  ];
    float imMz = fftMz[e+1];

    float Kxx = fftKxx[e];
    float Kyy = fftKyy[e];
    float Kzz = fftKzz[e];

    float Kyz = fftKyz[e];
    float Kxz = fftKxz[e];
    float Kxy = fftKxy[e];

    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;

  }

  //timer_stop("kernel_mul");
}

#ifdef __cplusplus
}
#endif
