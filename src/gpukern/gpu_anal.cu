#include "gpu_anal.h"
#include "../macros.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_anal_fw_step (float *minx, float *miny, float *minz, float *moutx, float *mouty, float *moutz, float *hx, float *hy, float *hz, float dt, float alpha, int N){
  
  int i = threadindex;
  if(i < N && (minx[i]!=0.0f || miny[i]!=0.0f || minz[i]!=0.0f) ){

  float hxy_r, hxyz_r;
  float rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot8;

  if (hx[i]==0.0f && hy[i] ==0.0f){
    rot0 = 0.0f;
    rot1 = 0.0f;
    rot2 = -1.0f;
    rot3 = 0.0f;
    rot4 = 1.0f;
    rot5 = 0.0f;
    rot6 = 1.0f;
//  rot7 = 0.0f;
    rot8 = 0.0f;

    hxyz_r = 1.0f/hz[i];
  }
  else{
    float temp = hx[i]*hx[i] + hy[i]*hy[i];
    hxy_r = rsqrtf(temp);
    hxyz_r = rsqrtf(temp + hz[i]*hz[i]);

    rot0 = hx[i]*hxyz_r;
    rot1 = - hy[i]*hxy_r;
    rot2 = - rot0*hz[i]*hxy_r;
    rot3 = hy[i]*hxyz_r;
    rot4 = hx[i]*hxy_r;
    rot5 = rot1*hz[i]*hxyz_r;
    rot6 = hz[i]*hxyz_r;
//  rot[7] = 0.0f;
    rot8 = hxyz_r/hxy_r;
  }

  float mx_rot = minx[i]*rot0 + miny[i]*rot3 + minz[i]*rot6;
  float my_rot = minx[i]*rot1 + miny[i]*rot4;
  float mz_rot = minx[i]*rot2 + miny[i]*rot5 + minz[i]*rot8;

  float qt = dt / (1+alpha*alpha);
  float aqt = alpha*qt;

  float ex, sn, cs, denom;
  ex = exp(aqt/hxyz_r);
  __sincosf(qt/hxyz_r, &sn, &cs);
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


void gpu_anal_fw_step(float dt, float alpha, int N, float *m_in, float *m_out, float *h){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);
  
//   timer_start("gpu_anal_fw_step");
  _gpu_anal_fw_step <<<gridSize, blockSize>>> (&m_in[X*N], &m_in[Y*N], &m_in[Z*N], &m_out[X*N], &m_out[Y*N], &m_out[Z*N], &h[X*N], &h[Y*N], &h[Z*N], dt, alpha, N);
  gpu_sync();
//   timer_stop("gpu_anal_fw_step");
  

  return;
}


__global__ void _gpu_anal_pc_meah_h (float *h1, float *h2, int N){
  
  int i = threadindex;

  if (i<N)
    h1[i] = 0.5f*(h1[i] + h2[i]);
  
  return;
}



void gpu_anal_pc_mean_h(float *h1, float *h2, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

//   timer_start("gpu_anal_pc_mean_h");
  _gpu_anal_pc_meah_h <<<gridSize, blockSize>>> (h1, h2, N);
  gpu_sync();
//   timer_stop("gpu_anal_pc_mean_h");

  return;
}


#ifdef __cplusplus
}
#endif