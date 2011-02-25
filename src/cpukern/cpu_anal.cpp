/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_anal.h"
#include "../macros.h"
#include <math.h>
#include "cpu_linalg.h"
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif




///@internal kernel
typedef struct{
  float *min, *mout, *h;
  float dt, alpha;
  int N;
} cpu_anal_fw_step_arg;

void cpu_anal_fw_step_t(int id){
  
  cpu_anal_fw_step_arg *arg = (cpu_anal_fw_step_arg *) func_arg;

  float *mx = arg->min + 0*arg->N;
  float *my = arg->min + 1*arg->N;
  float *mz = arg->min + 2*arg->N;
  float *hx = arg->h + 0*arg->N;
  float *hy = arg->h + 1*arg->N;
  float *hz = arg->h + 2*arg->N;
  float *moutx = arg->mout + 0*arg->N;
  float *mouty = arg->mout + 1*arg->N;
  float *moutz = arg->mout + 2*arg->N;

  float hxy_r, hxyz_r;
  float rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot8;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++){

    
/*    hx[i] += mx[i];
    hy[i] += my[i];
    hz[i] += mz[i];*/
    if (mx[i]==0.0f && my[i]==0.0f && mz[i]==0.0f)
      return;

    if (hx[i]==0.0f && hy[i] ==0.0f){
      rot0 = 0.0f;
      rot1 = 0.0f;
      rot2 = -1.0f;
      rot3 = 0.0f;
      rot4 = 1.0f;
      rot5 = 0.0f;
      rot6 = 1.0f;
  //      rot[7] = 0.0f;
      rot8 = 0.0f;

      hxyz_r = 1.0f/hz[i];
    }
    else{

      float temp = hx[i]*hx[i] + hy[i]*hy[i];
      hxy_r = 1.0f/sqrt(temp);
      hxyz_r = 1.0f/sqrt(temp + hz[i]*hz[i]);
//      printf("h_ampl = %e\n", 1.0/hxyz_r);
//       if (i == 100)
//         printf("h = %e, %e, %e\t m = %e, %e, %e\n", hx[i], hy[i], hz[i], mx[i], my[i], mz[i]);

      rot0 = hx[i]*hxyz_r;
      rot1 = - hy[i]*hxy_r;
      rot2 = - rot0*hz[i]*hxy_r;
      rot3 = hy[i]*hxyz_r;
      rot4 = hx[i]*hxy_r;
      rot5 = rot1*hz[i]*hxyz_r;
      rot6 = hz[i]*hxyz_r;
  //      rot[7] = 0.0f;
      rot8 = hxyz_r/hxy_r;
    }

    float mx_rot = mx[i]*rot0 + my[i]*rot3 + mz[i]*rot6;
    float my_rot = mx[i]*rot1 + my[i]*rot4;
    float mz_rot = mx[i]*rot2 + my[i]*rot5 + mz[i]*rot8;

    float at = arg->dt / (1+arg->alpha*arg->alpha);
    float act = arg->alpha*at;

//     printf("%e, %e\n", -at/hxyz_r, act/hxyz_r);
    hxyz_r = hxyz_r;
    float ex, sn, cs, denom;
    ex = exp(act/hxyz_r);
    sn = sinf(-at/hxyz_r);
    cs = cosf(-at/hxyz_r);
    denom = ex*(1.0f+mx_rot) + (1.0f-mx_rot)/ex;

    float mx_rotnw = (ex*(1.0f+mx_rot) - (1.0f-mx_rot)/ex)/denom;
    float my_rotnw = 2.0f*(my_rot*cs - mz_rot*sn)/denom;
    float mz_rotnw = 2.0f*(my_rot*sn + mz_rot*cs)/denom;

    moutx[i] = mx_rotnw*rot0 + my_rotnw*rot1 + mz_rotnw*rot2;
    mouty[i] = mx_rotnw*rot3 + my_rotnw*rot4 + mz_rotnw*rot5;
    moutz[i] = mx_rotnw*rot6 + mz_rotnw*rot8;
  }
  return;
}

void cpu_anal_fw_step(float dt, float alpha, int N, float *min, float *mout, float *h){

  cpu_anal_fw_step_arg args;
  args.min = min;
  args.mout = mout;
  args.h = h;
  args.dt = dt;
  args.alpha = alpha;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_anal_fw_step_t);

  return;
}


///@todo skip this function and replace it in thby cpu_linear_combination(h1, h2, 0.5, 0.5, N)
void cpu_anal_pc_mean_h(float *h1, float *h2, int N){

  cpu_linear_combination(h1, h2, 0.5, 0.5, N);

  return;
}




#ifdef __cplusplus
}
#endif
