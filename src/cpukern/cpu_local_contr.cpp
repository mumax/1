/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_local_contr.h"
#include "thread_functions.h"
#include "../macros.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct{
  float *hx, *hy, *hz;
  float hext_x, hext_y, hext_z;
  int N;
}cpu_add_external_field_arg;

void cpu_add_external_field_t(int id){
  
  cpu_add_external_field_arg *arg = (cpu_add_external_field_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i = start; i < stop; i++){
    arg->hx[i] += arg->hext_x;
    arg->hy[i] += arg->hext_y;
    arg->hz[i] += arg->hext_z;
  }

  return;
}

void cpu_add_external_field(float* hx, float* hy, float* hz,
                            float hext_x, float hext_y, float hext_z,
                            int N){

  cpu_add_external_field_arg args;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_external_field_t);
  
  return;
}


typedef struct{
  float *mx, *my, *mz, *hx, *hy, *hz;
  float hext_x, hext_y, hext_z, anisK, U0, U1, U2;
  int N;
}cpu_add_local_fields_uniaxial_arg;

void cpu_add_local_fields_uniaxial_t(int id){
  
  cpu_add_local_fields_uniaxial_arg *arg = (cpu_add_local_fields_uniaxial_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);
  
  for (int i=start; i<stop; i++){
    float mu = arg->mx[i] * arg->U0 + arg->my[i] * arg->U1 + arg->mz[i] * arg->U2;
    arg->hx[i] += arg->hext_x + arg->anisK * mu * arg->U0;
    arg->hy[i] += arg->hext_y + arg->anisK * mu * arg->U1;
    arg->hz[i] += arg->hext_z + arg->anisK * mu * arg->U2;
  }
  
  return;
}

void cpu_add_local_fields_uniaxial(float *mx, float *my, float *mz, float* hx, float* hy, float* hz,
                                   float hext_x, float hext_y, float hext_z, float anisK, float U0, float U1, float U2,
                                   int N){

  cpu_add_local_fields_uniaxial_arg args;
  args.mx = mx;
  args.my = my;
  args.mz = mz;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.anisK = anisK;
  args.U0 = U0;
  args.U1 = U1;
  args.U2 = U2;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_local_fields_uniaxial_t);

  return;
}



typedef struct{
  float *mx, *my, *mz, *hx, *hy, *hz;
  float hext_x, hext_y, hext_z, K0, K1, U0_0, U0_1, U0_2, U1_0, U1_1, U1_2, U2_0, U2_1, U2_2;
  int N;
}cpu_add_local_fields_cubic_arg;

void cpu_add_local_fields_cubic_t(int id){
  
  cpu_add_local_fields_cubic_arg *arg = (cpu_add_local_fields_cubic_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);
  
  for (int i=start; i<stop; i++){
        //projection of m on cubic anisotropy axes
      float a0 = arg->mx[i]*arg->U0_0 + arg->my[i]*arg->U0_1 + arg->mz[i]*arg->U0_2;
      float a1 = arg->mx[i]*arg->U1_0 + arg->my[i]*arg->U1_1 + arg->mz[i]*arg->U1_2;
      float a2 = arg->mx[i]*arg->U2_0 + arg->my[i]*arg->U2_1 + arg->mz[i]*arg->U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = arg->K0 * (a11+a22) * a0  +  arg->K1 * a0  *a11 * a22;
      float dphi_1 = arg->K0 * (a00+a22) * a1  +  arg->K1 * a00 *a1  * a22;
      float dphi_2 = arg->K0 * (a00+a11) * a2  +  arg->K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      arg->hx[i] += arg->hext_x - dphi_0*arg->U0_0 - dphi_1*arg->U1_0 - dphi_2*arg->U2_0;
      arg->hy[i] += arg->hext_y - dphi_0*arg->U0_1 - dphi_1*arg->U1_1 - dphi_2*arg->U2_1;
      arg->hz[i] += arg->hext_z - dphi_0*arg->U0_2 - dphi_1*arg->U1_2 - dphi_2*arg->U2_2;
  }
  
  return;
}

void cpu_add_local_fields_cubic(float *mx, float *my, float *mz, float* hx, float* hy, float* hz,
                                float hext_x, float hext_y, float hext_z, float K0, float K1, 
                                float U0_0, float U0_1, float U0_2,
                                float U1_0, float U1_1, float U1_2,
                                float U2_0, float U2_1, float U2_2, 
                                int N){

  cpu_add_local_fields_cubic_arg args;
  args.mx = mx;
  args.my = my;
  args.mz = mz;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.K0 = K0;
  args.K1 = K1;
  args.U0_0 = U0_0;
  args.U0_1 = U0_1;
  args.U0_2 = U0_2;
  args.U1_0 = U1_0;
  args.U1_1 = U1_1;
  args.U1_2 = U1_2;
  args.U2_0 = U2_0;
  args.U2_1 = U2_1;
  args.U2_2 = U2_2;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_local_fields_cubic_t);

  return;
}




void cpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){

  float *mx = &(m[0*N]);
  float *my = &(m[1*N]);
  float *mz = &(m[2*N]);

  float *hx = &(h[0*N]);
  float *hy = &(h[1*N]);
  float *hz = &(h[2*N]);
  
  switch (anisType){
    case ANIS_NONE:
      cpu_add_external_field(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
      break;
    case ANIS_UNIAXIAL:
      cpu_add_local_fields_uniaxial(mx, my, mz, hx, hy, hz, Hext[X], Hext[Y], Hext[Z],
                                    anisK[0], anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
    case ANIS_CUBIC:
    //printf("anis: K, u,: %f %f  %f,%f,%f  \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2],anisAxes[3],anisAxes[4],anisAxes[5],anisAxes[6],anisAxes[7],anisAxes[8]);
      cpu_add_local_fields_cubic(mx, my, mz, hx, hy, hz, Hext[X], Hext[Y], Hext[Z],
                                 anisK[0], anisK[1],
                                 anisAxes[0], anisAxes[1], anisAxes[2],
                                 anisAxes[3], anisAxes[4], anisAxes[5],
                                 anisAxes[6], anisAxes[7], anisAxes[8],
                                 N);
      break;
    default: abort();
  }

  return;
}



//-------------------------------


typedef struct{
  float *mx, *my, *mz, *hx, *hy, *hz, *phi;
  float hext_x, hext_y, hext_z;
  int N;
}cpu_add_external_field_H_and_phi_arg;

void cpu_add_external_field_H_and_phi_t(int id){
  
  cpu_add_external_field_H_and_phi_arg *arg = (cpu_add_external_field_H_and_phi_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i = start; i < stop; i++){
    arg->hx[i] += arg->hext_x;
    arg->hy[i] += arg->hext_y;
    arg->hz[i] += arg->hext_z;
    arg->phi[i] -= arg->mx[i]*arg->hext_x + arg->my[i]*arg->hext_y + arg->mz[i]*arg->hext_z;
  }

  return;
}

void cpu_add_external_field_H_and_phi(float *mx, float *my, float *mz,
                                      float *hx, float *hy, float *hz, float *phi,
                                      float hext_x, float hext_y, float hext_z,
                                      int N){

  cpu_add_external_field_H_and_phi_arg args;
  args.mx = mx;
  args.my = my;
  args.mz = mz;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.phi = phi;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_external_field_H_and_phi_t);
  
  return;
}


typedef struct{
  float *mx, *my, *mz, *hx, *hy, *hz, *phi;
  float hext_x, hext_y, hext_z, anisK, U0, U1, U2;
  int N;
}cpu_add_local_fields_uniaxial_H_and_phi_arg;

void cpu_add_local_fields_uniaxial_H_and_phi_t(int id){
  
  cpu_add_local_fields_uniaxial_H_and_phi_arg *arg = (cpu_add_local_fields_uniaxial_H_and_phi_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);
  
  for (int i=start; i<stop; i++){
    float mu = arg->mx[i] * arg->U0 + arg->my[i] * arg->U1 + arg->mz[i] * arg->U2;
    arg->hx[i] += arg->hext_x + arg->anisK * mu * arg->U0;
    arg->hy[i] += arg->hext_y + arg->anisK * mu * arg->U1;
    arg->hz[i] += arg->hext_z + arg->anisK * mu * arg->U2;
    arg->phi[i] -= arg->mx[i]*arg->hext_x + arg->my[i]*arg->hext_y + arg->mz[i]*arg->hext_z + 
                   0.5*arg->anisK*mu*(arg->mx[i]*arg->U0 + arg->my[i]*arg->U1 + arg->mz[i]*arg->U2);
  }
  
  return;
}

void cpu_add_local_fields_uniaxial_H_and_phi(float *mx, float *my, float *mz, float* hx, float* hy, float* hz, float *phi,
                                             float hext_x, float hext_y, float hext_z, float anisK, float U0, float U1, float U2,
                                             int N){

  cpu_add_local_fields_uniaxial_H_and_phi_arg args;
  args.mx = mx;
  args.my = my;
  args.mz = mz;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.phi = phi;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.anisK = anisK;
  args.U0 = U0;
  args.U1 = U1;
  args.U2 = U2;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_local_fields_uniaxial_H_and_phi_t);

  return;
}


typedef struct{
  float *mx, *my, *mz, *hx, *hy, *hz, *phi;
  float hext_x, hext_y, hext_z, K0, K1, U0_0, U0_1, U0_2, U1_0, U1_1, U1_2, U2_0, U2_1, U2_2;
  int N;
}cpu_add_local_fields_cubic_H_and_phi_arg;

void cpu_add_local_fields_cubic_H_and_phi_t(int id){
  
  cpu_add_local_fields_cubic_H_and_phi_arg *arg = (cpu_add_local_fields_cubic_H_and_phi_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);
  
  for (int i=start; i<stop; i++){
        //projection of m on cubic anisotropy axes
      float a0 = arg->mx[i]*arg->U0_0 + arg->my[i]*arg->U0_1 + arg->mz[i]*arg->U0_2;
      float a1 = arg->mx[i]*arg->U1_0 + arg->my[i]*arg->U1_1 + arg->mz[i]*arg->U1_2;
      float a2 = arg->mx[i]*arg->U2_0 + arg->my[i]*arg->U2_1 + arg->mz[i]*arg->U2_2;
      
      float a00 = a0*a0;
      float a11 = a1*a1;
      float a22 = a2*a2;
      
        // differentiated energy expressions
      float dphi_0 = arg->K0 * (a11+a22) * a0  +  arg->K1 * a0  *a11 * a22;
      float dphi_1 = arg->K0 * (a00+a22) * a1  +  arg->K1 * a00 *a1  * a22;
      float dphi_2 = arg->K0 * (a00+a11) * a2  +  arg->K1 * a00 *a11 * a2 ;
      
        // adding applied field and cubic axes contribution
      float cub0 = dphi_0*arg->U0_0 + dphi_1*arg->U1_0 + dphi_2*arg->U2_0;
      float cub1 = dphi_0*arg->U0_1 + dphi_1*arg->U1_1 + dphi_2*arg->U2_1;
      float cub2 = dphi_0*arg->U0_2 + dphi_1*arg->U1_2 + dphi_2*arg->U2_2;
      arg->hx[i] += arg->hext_x - cub0;
      arg->hy[i] += arg->hext_y - cub1;
      arg->hz[i] += arg->hext_z - cub2;
      arg->phi[i] -= arg->mx[i]*arg->hext_x + arg->my[i]*arg->hext_y + arg->mz[i]*arg->hext_z 
                - 0.25f*(arg->mx[i]*cub0 + arg->my[i]*cub1 + arg->mz[i]*cub2);
      
  }
  
  return;
}

void cpu_add_local_fields_cubic_H_and_phi(float *mx, float *my, float *mz, float* hx, float* hy, float* hz, float *phi,
                                float hext_x, float hext_y, float hext_z, float K0, float K1, 
                                float U0_0, float U0_1, float U0_2,
                                float U1_0, float U1_1, float U1_2,
                                float U2_0, float U2_1, float U2_2, 
                                int N){

  cpu_add_local_fields_cubic_H_and_phi_arg args;
  args.mx = mx;
  args.my = my;
  args.mz = mz;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.phi = phi;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.K0 = K0;
  args.K1 = K1;
  args.U0_0 = U0_0;
  args.U0_1 = U0_1;
  args.U0_2 = U0_2;
  args.U1_0 = U1_0;
  args.U1_1 = U1_1;
  args.U1_2 = U1_2;
  args.U2_0 = U2_0;
  args.U2_1 = U2_1;
  args.U2_2 = U2_2;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_local_fields_cubic_H_and_phi_t);

  return;
}




void cpu_add_local_fields_H_and_phi (float* m, float* h, float *phi, int N, float* Hext, int anisType, float* anisK, float* anisAxes){

  float *mx = &(m[0*N]);
  float *my = &(m[1*N]);
  float *mz = &(m[2*N]);

  float *hx = &(h[0*N]);
  float *hy = &(h[1*N]);
  float *hz = &(h[2*N]);
  
  switch (anisType){
    case ANIS_NONE:
      cpu_add_external_field_H_and_phi(mx, my, mz, hx, hy, hz, phi, Hext[X], Hext[Y], Hext[Z], N);
      break;
    case ANIS_UNIAXIAL:
      cpu_add_local_fields_uniaxial_H_and_phi(mx, my, mz, hx, hy, hz, phi, Hext[X], Hext[Y], Hext[Z],
                                              anisK[0], anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
    case ANIS_CUBIC:
    //printf("anis: K, u,: %f %f  %f,%f,%f  \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2],anisAxes[3],anisAxes[4],anisAxes[5],anisAxes[6],anisAxes[7],anisAxes[8]);
      cpu_add_local_fields_cubic_H_and_phi(mx, my, mz, hx, hy, hz, phi, Hext[X], Hext[Y], Hext[Z],
                                 anisK[0], anisK[1],
                                 anisAxes[0], anisAxes[1], anisAxes[2],
                                 anisAxes[3], anisAxes[4], anisAxes[5],
                                 anisAxes[6], anisAxes[7], anisAxes[8],
                                 N);
      break;
    default: abort();
  }

  return;
}




#ifdef __cplusplus
}
#endif
