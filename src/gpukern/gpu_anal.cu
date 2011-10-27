/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_anal.h"
#include "../macros.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16


__global__ void _gpu_anal_fw_step (float *minx, float *miny, float *minz, 
                                   float *moutx, float *mouty, float *moutz, 
                                   float *hx, float *hy, float *hz, 
                                   float dt, float alpha, int N){
  
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





__device__ void _derivative_x(float *mx, float *my, float *mz, 
                              float *diffmx, float *diffmy, float *diffmz,
                              float ux, int i, int j, int k, 
                              int N0, int N1, int N2){
  
  float mx1, my1, mz1, mx2, my2, mz2;
  if (i-1 >= 0){
    int idx = (i-1)*N1*N2 + j*N2 + k;
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } else {
    // How to handle edge cells?
    // * leaving the m value zero gives too big a gradient
    // * setting it to the central value gives the actual gradient / 2, should not hurt
    // * problem with nonuniform norm!! what if a neighbor has zero norm (but still lies in the box)?
    int idx = (i)*N1*N2 + j*N2 + k;
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  }
  if (i+1 < N0){
    int idx = (i+1)*N1*N2 + j*N2 + k;
    mx2 = mx[idx];
    my2 = my[idx];
    mz2 = mz[idx];
  } else {
    int idx = (i)*N1*N2 + j*N2 + k;
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } 
  *diffmx += ux * (mx2 - mx1);
  *diffmy += ux * (my2 - my1);
  *diffmz += ux * (mz2 - mz1);
  
  return;

}

__device__ void _derivative_y(float *mx, float *my, float *mz, 
                              float *diffmx, float *diffmy, float *diffmz,
                              float uy, int i, int j, int k, 
                              int N0, int N1, int N2){
  
  float mx1, my1, mz1, mx2, my2, mz2;
  if (j-1 >= 0){
    int idx = (i)*N1*N2 + (j-1)*N2 + k;
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } else {
    int idx = (i)*N1*N2 + (j)*N2 + k;
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } 
  if (j+1 < N1){
    int idx = (i)*N1*N2 + (j+1)*N2 + k;
    mx2 = mx[idx];
    my2 = my[idx];
    mz2 = mz[idx];
  } else {
    int idx = (i)*N1*N2 + (j)*N2 + k;
    mx2 = mx[idx];
    my2 = my[idx];
    mz2 = mz[idx];
  } 
  *diffmx += uy * (mx2 - mx1);
  *diffmy += uy * (my2 - my1);
  *diffmz += uy * (mz2 - mz1);

  return;
}

__device__ void _derivative_z(float *mx, float *my, float *mz, 
                              float *diffmx, float *diffmy, float *diffmz,
                              float uz, int i, int j, int k, 
                              int N0, int N1, int N2){

  float mx1, my1, mz1, mx2, my2, mz2;
  if (k-1 >= 0){
    int idx = (i)*N1*N2 + (j)*N2 + (k-1);
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } else {
    int idx = (i)*N1*N2 + (j)*N2 + (k);
    mx1 = mx[idx];
    my1 = my[idx];
    mz1 = mz[idx];
  } 
  if (k+1 < N2){
    int idx = (i)*N1*N2 + (j)*N2 + (k+1);
    mx2 = mx[idx];
    my2 = my[idx];
    mz2 = mz[idx];
  } else {
    int idx = (i)*N1*N2 + (j)*N2 + (k);
    mx2 = mx[idx];
    my2 = my[idx];
    mz2 = mz[idx];
  } 
  *diffmx += uz * (mx2 - mx1);
  *diffmy += uz * (my2 - my1);
  *diffmz += uz * (mz2 - mz1);

  return;
}
  


__global__ void   _gpu_anal_fw_step_spin_torque(float *minx, float *miny, float *minz, 
                                                float *moutx, float *mouty, float *moutz,
                                                float *hx, float *hy, float *hz,
                                                float alpha, float beta, float epsilon, 
                                                float ux, float uy, float uz, 
                                                int N0, int N1, int N2, float dt_gilb){
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  
  if ( j<N1 && k<N2 ){
    
    int ind =  j*N2 + k;
    for (int i=0; i<N0; i++){
      ind += N1*N2;
        //evaluation spin torque contribution
      float diffmx = 0.0f, diffmy = 0.0f, diffmz = 0.0f;
      if (ux!=0.0f)
        _derivative_x(minx, miny, minz, &diffmx, &diffmy, &diffmz, ux, i, j, k, N0, N1, N2);
      if (uy!=0.0f)
        _derivative_y(minx, miny, minz, &diffmx, &diffmy, &diffmz, uy, i, j, k, N0, N1, N2);
      if (uz!=0.0f)
        _derivative_z(minx, miny, minz, &diffmx, &diffmy, &diffmz, uz, i, j, k, N0, N1, N2);
      
        //definition Hd (damping) and Hp (precession)
      float Hdx = alpha*hx[ind] + beta*diffmx;
      float Hdy = alpha*hy[ind] + beta*diffmy;
      float Hdz = alpha*hz[ind] + beta*diffmz;
      float Hd_r = rsqrtf(Hdx*Hdx + Hdy*Hdy + Hdz*Hdz);
      float Hpx = hx[ind] + epsilon * diffmx;
      float Hpy = hy[ind] + epsilon * diffmy;
      float Hpz = hz[ind] + epsilon * diffmz;
           
        //definition of the local rotation matrix based on Hp (precession)
      float Hxy_r, Hp_r;
      float rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot8;
      if (Hpx==0.0f && Hpy ==0.0f){
        rot0 = 0.0f;
        rot1 = 0.0f;
        rot2 = -1.0f;
        rot3 = 0.0f;
        rot4 = 1.0f;
        rot5 = 0.0f;
        rot6 = 1.0f;
    //  rot7 = 0.0f;
        rot8 = 0.0f;

        Hp_r = 1.0f/Hpz;
      }
      else{
        
        float temp = Hpx*Hpx + Hpy*Hpy;
        Hxy_r = rsqrtf(temp);
        Hp_r = rsqrtf(temp + Hpz*Hpz);

        rot0 = Hpx*Hp_r;
        rot1 = - Hpy*Hxy_r;
        rot2 = - rot0*Hpz*Hxy_r;
        rot3 = Hpy*Hp_r;
        rot4 = Hpx*Hxy_r;
        rot5 = rot1*Hpz*Hp_r;
        rot6 = Hpz*Hp_r;
    //  rot[7] = 0.0f;
        rot8 = Hp_r/Hxy_r;
      }
      
        //local FW rotation of m
      float mx_rot = minx[ind]*rot0 + miny[ind]*rot3 + minz[ind]*rot6;
      float my_rot = minx[ind]*rot1 + miny[ind]*rot4;
      float mz_rot = minx[ind]*rot2 + miny[ind]*rot5 + minz[ind]*rot8;

        //determination sin, cos (damping) and exp (precession)
      float sn, cs;
      float ex = exp(dt_gilb/Hd_r);        // damping
      __sincosf(dt_gilb/Hp_r, &sn, &cs);
      float denom = ex*(1.0f+mx_rot) + (1.0f-mx_rot)/ex;
      
        //applying analytical formulae
      float mx_rotnw = (ex*(1.0f+mx_rot) - (1.0f-mx_rot)/ex)/denom;
      float my_rotnw = 2.0f*(my_rot*cs - mz_rot*sn)/denom;
      float mz_rotnw = 2.0f*(my_rot*sn + mz_rot*cs)/denom;

        //local BW rotation of m
      moutx[i] = mx_rotnw*rot0 + my_rotnw*rot1 + mz_rotnw*rot2;
      mouty[i] = mx_rotnw*rot3 + my_rotnw*rot4 + mz_rotnw*rot5;
      moutz[i] = mx_rotnw*rot6 + mz_rotnw*rot8;

    }
  } 

  return;
}

void gpu_anal_fw_step_spin_torque(float *m_in, float *m_out, float *h,
                                  float alpha, float beta, float epsilon,
                                  float *u, float dt_gilb, int *size){

  dim3 gridSize(divUp(size[Z], BLOCKSIZE), divUp(size[Y], BLOCKSIZE));
  dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
  
  int N = size[X]*size[Y]*size[Z];
  
//   timer_start("gpu_anal_fw_step_spin_torque");
  _gpu_anal_fw_step_spin_torque <<<gridSize, blockSize>>> (&m_in[X*N], &m_in[Y*N], &m_in[Z*N], 
                                                           &m_out[X*N], &m_out[Y*N], &m_out[Z*N], 
                                                           &h[X*N], &h[Y*N], &h[Z*N], 
                                                           alpha, beta, epsilon, u[X], u[Y], u[Z],
                                                           size[X], size[Y], size[Z], dt_gilb);
  gpu_sync();
//   timer_stop("gpu_anal_fw_step_spin_torque");

  return;
}

#ifdef __cplusplus
}
#endif
