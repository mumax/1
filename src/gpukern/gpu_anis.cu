/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_anis.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


__global__ void _gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                                        float* mx, float* my, float* mz,
                                        float* kxx, float* kyy, float* kzz,
                                        float* kyz, float* kxz, float* kxy,
                                        int N){
  int i = threadindex;

  if(i < N){
    float Mx = mx[i];
    float My = my[i];
    float Mz = mz[i];

    float Kxx = kxx[i];
    float Kyy = kyy[i];
    float Kzz = kzz[i];
    float Kyz = kyz[i];
    float Kxz = kxz[i];
    float Kxy = kxy[i];

    hx[i] += Mx * Kxx + My * Kxy + Mz * Kxz;
    hy[i] += Mx * Kxy + My * Kyy + Mz * Kyz;
    hz[i] += Mx * Kxz + My * Kyz + Mz * Kzz;
  }
}


void gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                            float* mx, float* my, float* mz,
                            float* kxx, float* kyy, float* kzz,
                            float* kyz, float* kxz, float* kxy,
                            int N){
  
}
                            

#ifdef __cplusplus
}
#endif
