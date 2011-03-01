/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_local_contr2.h"
#include "gpu_mem.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_add_local_fields_uniaxial(float* mx, float* my, float* mz,
                                              float* hx, float* hy, float* hz,
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


// __global__ void _gpu_add_local_fields_cubic(float* mx, float* my, float* mz,
//                                             float* hx, float* hy, float* hz,
//                                             float hext_x, float hext_y, float hext_z,
//                                             float K1, float K2, 
//                                             float U0_1, float U0_2, float U0_3,
//                                             float U1_1, float U1_2, float U1_3,
//                                             float U2_1, float U2_2, float U2_3,
//                                             int N){
//   
//   if(i<N){
//         //projection of m on cubic anisotropy axes
//       float a0 = mx[i]*p_dev->anisAxes[0] + my[i]*p_dev->anisAxes[1] + mz[i]*p_dev->anisAxes[2];
//       float a1 = mx[i]*p_dev->anisAxes[3] + my[i]*p_dev->anisAxes[4] + mz[i]*p_dev->anisAxes[5];
//       float a2 = mz[i]*p_dev->anisAxes[6] + my[i]*p_dev->anisAxes[7] + mz[i]*p_dev->anisAxes[8];
//       
//       float a00 = a0*a0;
//       float a11 = a1*a1;
//       float a22 = a2*a2;
//       
//         // differentiated energy expressions
//       float dphi_0 = p_dev->anisK[0] * (a11+a22) * a0  +  p_dev->anisK[1] * a0  *a11 * a22;
//       float dphi_1 = p_dev->anisK[0] * (a00+a22) * a1  +  p_dev->anisK[1] * a00 *a1  * a22;
//       float dphi_2 = p_dev->anisK[0] * (a00+a11) * a2  +  p_dev->anisK[1] * a00 *a11 * a2 ;
//       
//         // adding applied field and cubic axes contribution
//       hx[i] += Hax - dphi_0*p_dev->anisAxes[0] - dphi_1*p_dev->anisAxes[3] - dphi_2*p_dev->anisAxes[6];
//       hy[i] += Hay - dphi_0*p_dev->anisAxes[1] - dphi_1*p_dev->anisAxes[4] - dphi_2*p_dev->anisAxes[7];
//       hz[i] += Haz - dphi_0*p_dev->anisAxes[2] - dphi_1*p_dev->anisAxes[5] - dphi_2*p_dev->anisAxes[8];
//   }
//  
// }



__global__ void _gpu_add_external_field(float* hx, float* hy, float* hz,
                                        float hext_x, float hext_y, float hext_z,
                                        int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}


void gpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){


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
       _gpu_add_external_field<<<gridsize, blocksize>>>(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
       break;
    case ANIS_UNIAXIAL:
	  //printf("anis: K, u,: %f  %f,%f,%f  \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2]);
      _gpu_add_local_fields_uniaxial<<<gridsize, blocksize>>>(mx, my, mz,
                                                             hx, hy, hz,
                                                             Hext[X], Hext[Y], Hext[Z],
                                                             anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
  }
  
  gpu_sync();
  return;
}



__global__ void _gpu_add_local_fields_uniaxial_H_and_phi(float* mx, float* my, float* mz,
                                                         float* hx, float* hy, float* hz, float *phi,
                                                         float hext_x, float hext_y, float hext_z,
                                                         float k_x2, float U0, float U1, float U2,
                                                         int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i]  += hext_x + k_x2 * mu * U0;
    hy[i]  += hext_y + k_x2 * mu * U1;
    hz[i]  += hext_z + k_x2 * mu * U2;
    phi[i] -= mx[i]*hext_x + my[i]*hext_y + mz[i]*hext_z + 0.5*k_x2*mu*(mx[i]*U0 + my[i]*U1 + mz[i]*U2);
    
  }
  
  return;
}

__global__ void _gpu_add_external_field_H_and_phi(float *mx, float *my, float *mz, 
                                                  float* hx, float* hy, float* hz, float *phi,
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


void gpu_add_local_fields_H_and_phi (float* m, float* h, float *phi, int N, float* Hext, int anisType, float* anisK, float* anisAxes){


  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

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
       _gpu_add_external_field_H_and_phi<<<gridsize, blocksize>>>(mx, my, mz, hx, hy, hz, phi, Hext[X], Hext[Y], Hext[Z],  N);
       break;
    case ANIS_UNIAXIAL:
    //printf("anis: K, u,: %f  %f,%f,%f  \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2]);
      _gpu_add_local_fields_uniaxial_H_and_phi<<<gridsize, blocksize>>>(mx, my, mz,
                                                                        hx, hy, hz, phi,
                                                                        Hext[X], Hext[Y], Hext[Z],
                                                                        anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
  }
  
  gpu_sync();
  return;
}





#ifdef __cplusplus
}
#endif
