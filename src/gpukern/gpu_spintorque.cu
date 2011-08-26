/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_spintorque.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif


///@todo Not correct at the edges with a normmap!


/// 2D, plane per plane, i=plane index
__global__ void _gpu_spintorque_deltaM(float* mx, float* my, float* mz,
                                       float* hx, float* hy, float* hz,
                                       float alpha, float beta, float epsillon,
                                       float ux, float uy, float uz,
                                       float* jmapx, float* jmapy, float* jmapz,
                                       float dt_gilb,
                                       int N0, int N1, int N2, int i){

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1*N2 + j*N2 + k;
  
  if (j < N1 && k < N2){

	// space-dependent map
	if(jmapx != NULL){
		ux *= jmapx[I];
	}
	if(jmapy != NULL){
		uy *= jmapy[I];
	}
	if(jmapz != NULL){
		uz *= jmapz[I];
	}


    // (1) calculate the directional derivative of (mx, my mz) along (ux,uy,uz).
    // Result is (diffmx, diffmy, diffmz) (= Hspin)
    // (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)
    
    //float m0x = mx[i*N1*N2 + j*N2 + k];
    float mx1 = 0.f, mx2 = 0.f, my1 = 0.f, my2 = 0.f, mz1 = 0.f, mz2 = 0.f;

    // derivative in X direction
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
    float diffmx = ux * (mx2 - mx1);
    float diffmy = ux * (my2 - my1);
    float diffmz = ux * (mz2 - mz1);


    // derivative in Y direction
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
    diffmx += uy * (mx2 - mx1);
    diffmy += uy * (my2 - my1);
    diffmz += uy * (mz2 - mz1);


    // derivative in Z direction
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
    diffmx += uz * (mx2 - mx1);
    diffmy += uz * (my2 - my1);
    diffmz += uz * (mz2 - mz1);


    //(2) torque terms

    // H
    float Hx = hx[I];
    float Hy = hy[I];
    float Hz = hz[I];

    // m
    float Mx = mx[I], My = my[I], Mz = mz[I];

    // Hp (Hprecess) = H + epsillon Hspin
    float Hpx = Hx + epsillon * diffmx;
    float Hpy = Hy + epsillon * diffmy;
    float Hpz = Hz + epsillon * diffmz;
    
    // - m cross Hprecess
    float _mxHpx = -My * Hpz + Hpy * Mz;
    float _mxHpy =  Mx * Hpz - Hpx * Mz;
    float _mxHpz = -Mx * Hpy + Hpx * My;

    // Hd Hdamp = alpha*H + beta*Hspin
    float Hdx = alpha*Hx + beta*diffmx;
    float Hdy = alpha*Hy + beta*diffmy;
    float Hdz = alpha*Hz + beta*diffmz;

    // - m cross Hdamp
    float _mxHdx = -My * Hdz + Hdy * Mz;
    float _mxHdy =  Mx * Hdz - Hdx * Mz;
    float _mxHdz = -Mx * Hdy + Hdx * My;

    
    // - m cross (m cross Hd)
    float _mxmxHdx =  My * _mxHdz - _mxHdy * Mz;
    float _mxmxHdy = -Mx * _mxHdz + _mxHdx * Mz;
    float _mxmxHdz =  Mx * _mxHdy - _mxHdx * My;
    
    hx[I] = dt_gilb * (-_mxHpx + _mxmxHdx);
    hy[I] = dt_gilb * (-_mxHpy + _mxmxHdy);
    hz[I] = dt_gilb * (-_mxHpz + _mxmxHdz);
  }
  
}


#define BLOCKSIZE 16

void gpu_spintorque_deltaM(float* m, float* h, float alpha, float beta, float epsillon, float* u, float* jmap, float dt_gilb, int N0,  int N1, int N2){

  dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  int N = N0 * N1 * N2;

  float* jmapx = NULL;
  float* jmapy = NULL;
  float* jmapz = NULL;

  if jmap != NULL{
  	float* jmapx = &jmap[0*N];
  	float* jmapy = &jmap[1*N];
  	float* jmapz = &jmap[2*N];
  }
  
  for(int i=0; i<N0; i++){
    _gpu_spintorque_deltaM<<<gridsize, blocksize>>>(&m[0*N], &m[1*N], &m[2*N], &h[0*N], &h[1*N], &h[2*N], alpha, beta, epsillon, u[0], u[1], u[2], jmapx,  jmapy, jmapz, dt_gilb, N0, N1, N2, i);
  }
  gpu_sync();
}





#define BLOCKSIZE 16

///@todo this is a slowish test implementation, use shared memory to avoid multiple reads (at least per plane)
__global__ void _gpu_directional_diff2D(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2, int i){

//int i = i;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < N1 && k < N2){

    float m0 = in[i*N1*N2 + j*N2 + k];
    float mx1, mx2;
    
    if (i-1 >= 0){ mx1 = in[(i-1)*N1*N2 + j*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (i+1 < N0){ mx2 = in[(i+1)*N1*N2 + j*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    float answer =  ux * (mx2 - mx1);

    if (j-1 >= 0){ mx1 = in[(i)*N1*N2 + (j-1)*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (j+1 < N1){ mx2 = in[(i)*N1*N2 + (j+1)*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    answer += uy * (mx2 - mx1);

    if (k-1 >= 0){ mx1 = in[(i)*N1*N2 + (j)*N2 + (k-1)]; } else { mx1 = m0; } // not 100% accurate
    if (k+1 < N2){ mx2 = in[(i)*N1*N2 + (j)*N2 + (k+1)]; } else { mx2 = m0; } // not 100% accurate
    answer += uz * (mx2 - mx1);

    out[i*N1*N2 + j*N2 + k] = answer;
  }
}


void gpu_directional_diff2D_async(float ux, float uy, float uz, float *input, float *output, int N0, int N1, int N2, int i){
    dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_directional_diff2D<<<gridsize, blocksize>>>(ux, uy, uz, input, output, N0, N1, N2, i);
}

void gpu_directionial_diff(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2){
  for(int i=0; i<N0; i++){
    gpu_directional_diff2D_async(ux, uy, uz, &in[i*N1*N2], &out[i*N1*N2], N0, N1, N2, i);
  }
  gpu_sync();
}


#ifdef __cplusplus
}
#endif
