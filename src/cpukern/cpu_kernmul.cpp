#include "cpu_kernmul.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_extract_real(float* complex, float* real, int NReal){
  #pragma omp parallel for
  for(int e=0; e<NReal; e++){
    real[e] = complex[2*e];
  }
}

void cpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers){

  //timer_start("kernel_mul");
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  #pragma omp parallel for
  for(int e=0; e<nRealNumbers; e+=2){
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

  //timer_stop("kernel_mul");
}

#ifdef __cplusplus
}
#endif
