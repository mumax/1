/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_exch.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 2D, plane per plane, i=plane index
__global__ void _gpu_add_exch6(float* mx, float* my, float* mz,
                               float* hx, float* hy, float* hz,
                               int N0, int N1, int N2,
                               int wrap0, int wrap1, int wrap2,
                               float fac0, float fac1, float fac2, 
                               int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1*N2 + j*N2 + k; // linear array index
  
  if (j < N1 && k < N2){

    // Local H initiated
	float Hx = hx[I];
	float Hy = hy[I];
	float Hz = hz[I];

	float mx1, my1, mz1; // magnetization of neighbor 1
	float mx2, my2, mz2; // magnetization of neighbor 2
	
	// Now add Neighbors.

    // neighbors in X direction
	int idx;
    if (i-1 >= 0){
      idx = (i-1)*N1*N2 + j*N2 + k;
    } else {
		if(wrap0){
			idx = (N0-1)*N1*N2 + j*N2 + k;
		}else{
      		idx = I;
		}
    }
	mx1 = mx[idx]; my1 = my[idx]; mz1 = mz[idx];

 	if (i+1 < N0){
      idx = (i+1)*N1*N2 + j*N2 + k;
    } else {
		if(wrap0){
			idx = (0)*N1*N2 + j*N2 + k;
		}else{
      		idx = I;
		}
    } 
	mx2 = mx[idx]; my2 = my[idx]; mz2 = mz[idx];

    Hx += fac0 * (mx1 + mx2 - 2.0f*mx[I]);
    Hy += fac0 * (my1 + my2 - 2.0f*my[I]);
    Hz += fac0 * (mz1 + mz2 - 2.0f*mz[I]);

    // neighbors in Y direction
    if (j-1 >= 0){
      idx = i*N1*N2 + (j-1)*N2 + k;
    } else {
		if(wrap1){
			idx = i*N1*N2 + (N1-1)*N2 + k;
		}else{
      		idx = I;
		}
    }
	mx1 = mx[idx]; my1 = my[idx]; mz1 = mz[idx];

 	if (j+1 < N1){
      idx = i*N1*N2 + (j+1)*N2 + k;
    } else {
		if(wrap1){
			idx = i*N1*N2 + (0)*N2 + k;
		}else{
      		idx = I;
		}
    } 
	mx2 = mx[idx]; my2 = my[idx]; mz2 = mz[idx];

    Hx += fac1 * (mx1 + mx2 - 2.0f*mx[I]);
    Hy += fac1 * (my1 + my2 - 2.0f*my[I]);
    Hz += fac1 * (mz1 + mz2 - 2.0f*mz[I]);

    // neighbors in Z direction
    if (k-1 >= 0){
      idx = i*N1*N2 + j*N2 + (k-1);
    } else {
		if(wrap2){
			idx = i*N1*N2 + j*N2 + (N2-1);
		}else{
      		idx = I;
		}
    }
	mx1 = mx[idx]; my1 = my[idx]; mz1 = mz[idx];

 	if (k+1 < N2){
      idx =  i*N1*N2 + j*N2 + (k+1);
    } else {
		if(wrap2){
			idx = i*N1*N2 + j*N2 + (0);
		}else{
      		idx = I;
		}
    } 
	mx2 = mx[idx]; my2 = my[idx]; mz2 = mz[idx];
   
    Hx += fac2 * (mx1 + mx2 - 2.0f*mx[I]);
    Hy += fac2 * (my1 + my2 - 2.0f*my[I]);
    Hz += fac2 * (mz1 + mz2 - 2.0f*mz[I]);

	// Write back to global memory
    hx[I] = Hx;
    hy[I] = Hy;
    hz[I] = Hz;
  }
  
}


#define BLOCKSIZE 16
void gpu_add_exch(float* m, float* h, int *size, int *periodic, int *exchInConv, float *cellSize, int type){
  assert(type == EXCH_6NGBR);
  dim3 gridsize(divUp(size[Y], BLOCKSIZE), divUp(size[Z], BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  int N = size[X]*size[Y]*size[Z];

  float fac0 = 1.0f/(cellSize[0] * cellSize[0]);
  float fac1 = 1.0f/(cellSize[1] * cellSize[1]);
  float fac2 = 1.0f/(cellSize[2] * cellSize[2]);

  for(int i=0; i<size[X]; i++){
    _gpu_add_exch6<<<gridsize, blocksize>>>(&m[0*N], &m[1*N], &m[2*N], &h[0*N], &h[1*N], &h[2*N], size[X], size[Y], size[Z], periodic[X], periodic[Y], periodic[Z], fac0, fac1, fac2, i);
  }
  gpu_sync();
}



#ifdef __cplusplus
}
#endif
