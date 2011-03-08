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
                                    anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
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
    default: abort();
  }

  return;
}




#ifdef __cplusplus
}
#endif
