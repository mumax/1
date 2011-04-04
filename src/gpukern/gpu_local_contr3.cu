/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_local_contr3.h"
#include "gpu_mem.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_add_uniaxial_anis_uniform_Hext(float *mx, float *my, float *mz,
                                                    float *hx, float *hy, float *hz,
                                                    float hext_x, float hext_y, float hext_z,
                                                    float k_x2, float U0, float U1, float U2,
                                                    int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;
    hx[i] += hext_x + k_x2 * mu * U0;
    hy[i] += hext_y + k_x2 * mu * U1;
    hz[i] += hext_z + k_x2 * mu * U2;
  }
  
  return;
}


__global__ void _gpu_add_uniaxial_anis_non_uniform_Hext(float *mx, float *my, float *mz,
                                                        float *hx, float *hy, float *hz,
                                                        float *hext_x, float *hext_y, float *hext_z,
                                                        float k_x2, float U0, float U1, float U2,
                                                        int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;
    hx[i] += hext_x[i] + k_x2 * mu * U0;
    hy[i] += hext_y[i] + k_x2 * mu * U1;
    hz[i] += hext_z[i] + k_x2 * mu * U2;
  }
  
  return;
}


__global__ void _gpu_add_cubic_anis_uniform_Hext(float *mx, float *my, float *mz,
                                                 float *hx, float *hy, float *hz,
                                                 float hext_x, float hext_y, float hext_z,
                                                 float K0, float K1, 
                                                 float U0_0, float U0_1, float U0_2,
                                                 float U1_0, float U1_1, float U1_2,
                                                 float U2_0, float U2_1, float U2_2,
                                                 int N){
  
  int i = threadindex;

  if(i<N){
        //projection of m on cubic anisotropy axes
      float a0 = mx[i]*U0_0 + my[i]*U0_1 + mz[i]*U0_2;
      float a1 = mx[i]*U1_0 + my[i]*U1_1 + mz[i]*U1_2;
      float a2 = mx[i]*U2_0 + my[i]*U2_1 + mz[i]*U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = K0 * (a11+a22) * a0  +  K1 * a0  *a11 * a22;
      float dphi_1 = K0 * (a00+a22) * a1  +  K1 * a00 *a1  * a22;
      float dphi_2 = K0 * (a00+a11) * a2  +  K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      hx[i] += hext_x - dphi_0*U0_0 - dphi_1*U1_0 - dphi_2*U2_0;
      hy[i] += hext_y - dphi_0*U0_1 - dphi_1*U1_1 - dphi_2*U2_1;
      hz[i] += hext_z - dphi_0*U0_2 - dphi_1*U1_2 - dphi_2*U2_2;
  }
  
  return;
 
}


__global__ void _gpu_add_cubic_anis_non_uniform_Hext(float *mx, float *my, float *mz,
                                                     float *hx, float *hy, float *hz,
                                                     float *hext_x, float *hext_y, float *hext_z,
                                                     float K0, float K1, 
                                                     float U0_0, float U0_1, float U0_2,
                                                     float U1_0, float U1_1, float U1_2,
                                                     float U2_0, float U2_1, float U2_2,
                                                     int N){
  
  int i = threadindex;

  if(i<N){
        //projection of m on cubic anisotropy axes
      float a0 = mx[i]*U0_0 + my[i]*U0_1 + mz[i]*U0_2;
      float a1 = mx[i]*U1_0 + my[i]*U1_1 + mz[i]*U1_2;
      float a2 = mx[i]*U2_0 + my[i]*U2_1 + mz[i]*U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = K0 * (a11+a22) * a0  +  K1 * a0  *a11 * a22;
      float dphi_1 = K0 * (a00+a22) * a1  +  K1 * a00 *a1  * a22;
      float dphi_2 = K0 * (a00+a11) * a2  +  K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      hx[i] += hext_x[i] - dphi_0*U0_0 - dphi_1*U1_0 - dphi_2*U2_0;
      hy[i] += hext_y[i] - dphi_0*U0_1 - dphi_1*U1_1 - dphi_2*U2_1;
      hz[i] += hext_z[i] - dphi_0*U0_2 - dphi_1*U1_2 - dphi_2*U2_2;
  }
  
  return;
 
}


__global__ void _gpu_add_uniform_Hext(float *hx, float *hy, float *hz,
                              float hext_x, float hext_y, float hext_z,
                              int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}

__global__ void _gpu_add_non_uniform_Hext(float *hx, float *hy, float *hz,
                                          float *hext_x, float *hext_y, float *hext_z,
                                          int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x[i];
    hy [i] += hext_y[i];
    hz [i] += hext_z[i];
  }
}


void gpu_add_local_fields (float *m, float *h, int N, int HextType, float *Hext, int anisType, float *anisK, float *anisAxes){


  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  /*
    Uniaxial anisotropy:
    H_anis = ( 2*K_1 / (mu0 Ms) )  ( m . u ) u
	u = axis, normalized
  */
  
  dim3 gridsize, blocksize;
  make1dconf(N, &gridsize, &blocksize);

  switch (anisType){
    default: abort();
    case ANIS_NONE:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_uniform_Hext<<<gridsize, blocksize>>>(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
      else
        _gpu_add_non_uniform_Hext<<<gridsize, blocksize>>>(hx, hy, hz,  &Hext[X*N], &Hext[Y*N], Hext[&Z*N],  N);
      break;
    case ANIS_UNIAXIAL:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_uniaxial_anis_uniform_Hext<<<gridsize, blocksize>>>(mx, my, mz,
                                                                hx, hy, hz,
                                                                &Hext[X*N], &Hext[Y*N], &Hext[Z*N],
                                                                anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
      else
         _gpu_add_uniaxial_anis_non_uniform_Hext<<<gridsize, blocksize>>>(mx, my, mz,
                                                                                         hx, hy, hz,
                                                                                         &Hext[X*N], &Hext[Y*N], &Hext[Z*N],
                                                                                         anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
        
      break;
    case ANIS_CUBIC:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_cubic_anis_uniform_Hext<<<gridsize, blocksize>>>(mx, my, mz,
                                                                  hx, hy, hz,
                                                                  Hext[X], Hext[Y], Hext[Z],
                                                                  anisK[0], anisK[1],
                                                                  anisAxes[0], anisAxes[1], anisAxes[2],
                                                                  anisAxes[3], anisAxes[4], anisAxes[5],
                                                                  anisAxes[6], anisAxes[7], anisAxes[8],
                                                                  N);
      else
        _gpu_add_cubic_anis_non_uniform_Hext<<<gridsize, blocksize>>>(mx, my, mz,
                                                                      hx, hy, hz,
                                                                      &Hext[X*N], &Hext[Y*N], &Hext[Z*N],
                                                                      anisK[0], anisK[1],
                                                                      anisAxes[0], anisAxes[1], anisAxes[2],
                                                                      anisAxes[3], anisAxes[4], anisAxes[5],
                                                                      anisAxes[6], anisAxes[7], anisAxes[8],
                                                                      N);
      break;
  }
  
  gpu_sync();
  return;
}








__global__ void _gpu_add_uniaxial_anis_uniform_Hext_and_phi(float *mx, float *my, float *mz,
                                                            float *hx, float *hy, float *hz, float *phi,
                                                            float hext_x, float hext_y, float hext_z,
                                                            float k_x2, float U0, float U1, float U2,
                                                            int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i]  += hext_x + k_x2 * mu * U0;
    hy[i]  += hext_y + k_x2 * mu * U1;
    hz[i]  += hext_z + k_x2 * mu * U2;
    phi[i] -= mx[i]*hext_x + my[i]*hext_y + mz[i]*hext_z + 0.5f*k_x2*mu*(mx[i]*U0 + my[i]*U1 + mz[i]*U2);
    
  }
  
  return;
}


__global__ void _gpu_add_uniaxial_anis_non_uniform_Hext_and_phi(float *mx, float *my, float *mz,
                                                                float *hx, float *hy, float *hz, float *phi,
                                                                float *hext_x, float *hext_y, float *hext_z,
                                                                float k_x2, float U0, float U1, float U2,
                                                                int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i]  += hext_x[i] + k_x2 * mu * U0;
    hy[i]  += hext_y[i] + k_x2 * mu * U1;
    hz[i]  += hext_z[i] + k_x2 * mu * U2;
    phi[i] -= mx[i]*hext_x[i] + my[i]*hext_y[i] + mz[i]*hext_z[i] + 0.5f*k_x2*mu*(mx[i]*U0 + my[i]*U1 + mz[i]*U2);
    
  }
  
  return;
}

__global__ void _gpu_add_cubic_anis_uniform_Hext_and_phi(float *mx, float *my, float *mz,
                                                         float *hx, float *hy, float *hz, float *phi,
                                                         float hext_x, float hext_y, float Hext_z,
                                                         float K0, float K1, 
                                                         float U0_0, float U0_1, float U0_2,
                                                         float U1_0, float U1_1, float U1_2,
                                                         float U2_0, float U2_1, float U2_2,
                                                         int N){
  
  int i = threadindex;

  if(i<N){
        //projection of m on cubic anisotropy axes
      float a0 = mx[i]*U0_0 + my[i]*U0_1 + mz[i]*U0_2;
      float a1 = mx[i]*U1_0 + my[i]*U1_1 + mz[i]*U1_2;
      float a2 = mx[i]*U2_0 + my[i]*U2_1 + mz[i]*U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = K0 * (a11+a22) * a0  +  K1 * a0  *a11 * a22;
      float dphi_1 = K0 * (a00+a22) * a1  +  K1 * a00 *a1  * a22;
      float dphi_2 = K0 * (a00+a11) * a2  +  K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      float cub0 = dphi_0*U0_0 + dphi_1*U1_0 + dphi_2*U2_0;
      float cub1 = dphi_0*U0_1 + dphi_1*U1_1 + dphi_2*U2_1;
      float cub2 = dphi_0*U0_2 + dphi_1*U1_2 + dphi_2*U2_2;
      hx[i] += hext_x - cub0;
      hy[i] += hext_y - cub1;
      hz[i] += hext_z - cub2;
      phi[i] -= mx[i]*hext_x + my[i]*hext_y + mz[i]*hext_z 
                - 0.25f*(mx[i]*cub0 + my[i]*cub1 + mz[i]*cub2);

  }
  
  return;
 
}

__global__ void _gpu_add_cubic_anis_non_uniform_Hext_and_phi(float *mx, float *my, float *mz,
                                                             float *hx, float *hy, float *hz, float *phi,
                                                             float *hext_x, float *hext_y, float *hext_z,
                                                             float K0, float K1, 
                                                             float U0_0, float U0_1, float U0_2,
                                                             float U1_0, float U1_1, float U1_2,
                                                             float U2_0, float U2_1, float U2_2,
                                                             int N){
      
  int i = threadindex;

  if(i<N){
        //projection of m on cubic anisotropy axes
      float a0 = mx[i]*U0_0 + my[i]*U0_1 + mz[i]*U0_2;
      float a1 = mx[i]*U1_0 + my[i]*U1_1 + mz[i]*U1_2;
      float a2 = mx[i]*U2_0 + my[i]*U2_1 + mz[i]*U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = K0 * (a11+a22) * a0  +  K1 * a0  *a11 * a22;
      float dphi_1 = K0 * (a00+a22) * a1  +  K1 * a00 *a1  * a22;
      float dphi_2 = K0 * (a00+a11) * a2  +  K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      float cub0 = dphi_0*U0_0 + dphi_1*U1_0 + dphi_2*U2_0;
      float cub1 = dphi_0*U0_1 + dphi_1*U1_1 + dphi_2*U2_1;
      float cub2 = dphi_0*U0_2 + dphi_1*U1_2 + dphi_2*U2_2;
      hx[i] += hext_x[i] - cub0;
      hy[i] += hext_y[i] - cub1;
      hz[i] += hext_z[i] - cub2;
      phi[i] -= mx[i]*hext_x[i] + my[i]*hext_y[i] + mz[i]*hext_z[i] 
                - 0.25f*(mx[i]*cub0 + my[i]*cub1 + mz[i]*cub2);

  }
  
  return;
 
}

__global__ void _gpu_add_uniform_Hext_and_phi(float *mx, float *my, float *mz, 
                                              float *hx, float *hy, float *hz, float *phi,
                                              float hext_x, float hext_y, float hext_z,
                                              int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
    phi[i] -= mx[i]*hext_x + my[i]*hext_y + mz[i]*hext_z;
  }
}

__global__ void _gpu_add_non_uniform_Hext_and_phi(float *mx, float *my, float *mz, 
                                                  float *hx, float *hy, float *hz, float *phi,
                                                  float *hext_x, float *hext_y, float *hext_z,
                                                  int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x[i];
    hy [i] += hext_y[i];
    hz [i] += hext_z[i];
    phi[i] -= mx[i]*hext_x[i] + my[i]*hext_y[i] + mz[i]*hext_z[i];
  }
}

void gpu_add_local_fields_H_and_phi (float* m, float* h, float *phi, int N, int HextType, float* Hext, int anisType, float* anisK, float* anisAxes){


  float *mx = &(m[0*N]);
  float *my = &(m[1*N]);
  float *mz = &(m[2*N]);

  float *hx = &(h[0*N]);
  float *hy = &(h[1*N]);
  float *hz = &(h[2*N]);

  /*
    Uniaxial anisotropy:
    H_anis = (2*K_1 / (mu0 Ms) )  ( m . u ) u
  u = axis, normalized
  */
  
  dim3 gridsize, blocksize;
  make1dconf(N, &gridsize, &blocksize);

  switch (anisType){
    default: abort();
    case ANIS_NONE:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz, hx, hy, hz, phi, Hext[X], Hext[Y], Hext[Z],  N);
      else
        _gpu_add_non_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz, hx, hy, hz, phi, &Hext[X*N], &Hext[Y*N], &Hext[Z*N],  N);
      break;
    case ANIS_UNIAXIAL:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_uniaxial_anis_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz,
                                                                             hx, hy, hz, phi,
                                                                             Hext[X], Hext[Y], Hext[Z],
                                                                             anisK[0], anisAxes[0], anisAxes[1], anisAxes[2], N);
      else
        _gpu_add_uniaxial_anis_non_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz,
                                                                                 hx, hy, hz, phi,
                                                                                 &Hext[X*N], &Hext[Y*N], &Hext[Z*N],
                                                                                 anisK[0], anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
    case ANIS_CUBIC:
      if (HextType == HEXT_UNIFORM)
        _gpu_add_cubic_anis_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz,
                                                                          hx, hy, hz, phi,
                                                                          Hext[X], Hext[Y], Hext[Z],
                                                                          anisK[0], anisK[1],
                                                                          anisAxes[0], anisAxes[1], anisAxes[2],
                                                                          anisAxes[3], anisAxes[4], anisAxes[5],
                                                                          anisAxes[6], anisAxes[7], anisAxes[8],
                                                                          N);
     else
        _gpu_add_cubic_anis_non_uniform_Hext_and_phi<<<gridsize, blocksize>>>(mx, my, mz,
                                                                              hx, hy, hz, phi,
                                                                              &Hext[X*N], &Hext[Y*N], &Hext[Z*N],
                                                                              anisK[0], anisK[1],
                                                                              anisAxes[0], anisAxes[1], anisAxes[2],
                                                                              anisAxes[3], anisAxes[4], anisAxes[5],
                                                                              anisAxes[6], anisAxes[7], anisAxes[8],
                                                                              N);
     break;
  }
  
  gpu_sync();
  return;
}





#ifdef __cplusplus
}
#endif
