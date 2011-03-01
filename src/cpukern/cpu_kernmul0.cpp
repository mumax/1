/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_kernmul.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

// void cpu_extract_real(float* complex, float* real, int NReal){
//   #pragma omp parallel for
//   for(int e=0; e<NReal; e++){
//     real[e] = complex[2*e];
//   }
// }

void cpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  #pragma omp parallel for
  for(int i=0; i<nRealNumbers/2; i++){
    int e = i * 2;
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


void cpu_kernelmul4(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz,
                    int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  #pragma omp parallel for
  for(int i=0; i<nRealNumbers/2; i++){
    int e = i * 2;
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

    fftMx[e  ] = reMx * Kxx;
    fftMx[e+1] = imMx * Kxx;
    fftMy[e  ] = reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMy * Kyz + imMz * Kzz;
  }
}

#ifdef __cplusplus
}
#endif
