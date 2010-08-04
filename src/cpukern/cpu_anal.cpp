#include "cpu_anal.h"
#include "../macros.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

///@internal kernel
 void _cpu_anal_fw_step (float *mx, float *my, float *mz, float *hx, float *hy, float *hz, float dt, float alpha, int i){

//   int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float hxy_r, hxyz_r;
  float rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot8;

//  if (mx[i]==0.0f && my[i]==0.0f && *mz[i]==0.0f)
//    continue;

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

/// @todo check used parameters due to normalization of constants!!
  float at = dt / (1+alpha*alpha);
  float act = alpha*at;
// ----------------------------------------

  float ex, sn, cs, denom;
  ex = exp(act/hxyz_r);
  __sincosf(at/hxyz_r, &sn, &cs);
  denom = ex*(1.0f+mx_rot) + (1.0f-mx_rot)/ex;

  float mx_rotnw = (ex*(1.0f+mx_rot) - (1.0f-mx_rot)/ex)/denom;
  float my_rotnw = 2.0f*(my_rot*cs - mz_rot*sn)/denom;
  float mz_rotnw = 2.0f*(my_rot*sn + mz_rot*cs)/denom;

// in deze lijnen komt fout tot uiting
  mx[i] = mx_rotnw*rot0 + my_rotnw*rot1 + mz_rotnw*rot2;
  my[i] = mx_rotnw*rot3 + my_rotnw*rot4 + mz_rotnw*rot5;
  mz[i] = mx_rotnw*rot6 + mz_rotnw*rot8;
// -----------------------------------


  return;
}


void cpu_anal_fw_step_unsafe(float* m, float* h, float dt, float alpha, int N){

//   int gridSize = -1, blockSize = -1;
//   make1dconf(N, &gridSize, &blockSize);

  //timer_start("cpu_anal_fw_step");
 for(int i=0; i<N; i++){
   ///@todo inline this function
    _cpu_anal_fw_step(&m[X*N], &m[Y*N], &m[Z*N], &h[X*N], &h[Y*N], &h[Z*N], dt, alpha, i);
  }
  
  //timer_stop("cpu_anal_fw_step");

}

#ifdef __cplusplus
}
#endif
