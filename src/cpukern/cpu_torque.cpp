/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_torque.h"
#include "thread_functions.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct{
  float *m, *h;
  float alpha, dt_gilb;
  int N;
} cpu_deltaM_arg;


void cpu_deltaM_t(int id){

  cpu_deltaM_arg *arg = (cpu_deltaM_arg *) func_arg;

  float* mx = &(arg->m[0*arg->N]);
  float* my = &(arg->m[1*arg->N]);
  float* mz = &(arg->m[2*arg->N]);

  float* hx = &(arg->h[0*arg->N]);
  float* hy = &(arg->h[1*arg->N]);
  float* hz = &(arg->h[2*arg->N]);
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++){
    // - m cross H
    float _mxHx =  my[i] * hz[i] - hy[i] * mz[i];
    float _mxHy = -mx[i] * hz[i] + hx[i] * mz[i];
    float _mxHz =  mx[i] * hy[i] - hx[i] * my[i];
    
    // - m cross (m cross H)
    float _mxmxHx = -my[i] * _mxHz + _mxHy * mz[i];
    float _mxmxHy =  mx[i] * _mxHz - _mxHx * mz[i];
    float _mxmxHz = -mx[i] * _mxHy + _mxHx * my[i];
    
    hx[i] = arg->dt_gilb * (_mxHx + _mxmxHx * arg->alpha);
    hy[i] = arg->dt_gilb * (_mxHy + _mxmxHy * arg->alpha);
    hz[i] = arg->dt_gilb * (_mxHz + _mxmxHz * arg->alpha);
  }

  return;
}

void cpu_deltaM(float* m, float* h, float alpha_mul, float* alpha_map, float dt_gilb, int N){

  assert(alpha_map == NULL); // space-dependent alpha not yet implemented

  cpu_deltaM_arg args;
  args.m = m;
  args.h = h;
  args.alpha = alpha_mul;
  args.dt_gilb = dt_gilb;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_deltaM_t);

  return;
}


#ifdef __cplusplus
}
#endif
