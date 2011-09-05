/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_torque.h"
#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @internal
__global__ void _gpu_deltaM(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float alpha_mul, float* alpha_map, float dt_gilb, int N){

  int i = threadindex;

  if(i < N){
	float alpha;
	if(alpha_map == NULL){
		alpha = alpha_mul;	
	}else{
		alpha = alpha_mul * alpha_map[i];
	}
	
    float Mx = mx[i];
    float My = my[i];
    float Mz = mz[i];
    
    float Hx = hx[i];
    float Hy = hy[i];
    float Hz = hz[i];
    
    //  m cross H
    float _mxHx =  My * Hz - Hy * Mz;
    float _mxHy = -Mx * Hz + Hx * Mz;
    float _mxHz =  Mx * Hy - Hx * My;

    // - m cross (m cross H)
    float _mxmxHx = -My * _mxHz + _mxHy * Mz;
    float _mxmxHy = +Mx * _mxHz - _mxHx * Mz;
    float _mxmxHz = -Mx * _mxHy + _mxHx * My;

    hx[i] = dt_gilb * (_mxHx + _mxmxHx * alpha);
    hy[i] = dt_gilb * (_mxHy + _mxmxHy * alpha);
    hz[i] = dt_gilb * (_mxHz + _mxmxHz * alpha);
  }
}

void gpu_deltaM(float* m, float* h, float alpha_mul, float* alpha_map, float dt_gilb, int N){

  dim3 gridSize, blockSize;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  _gpu_deltaM<<<gridSize, blockSize>>>(mx, my, mz, hx, hy, hz, alpha_mul, alpha_map, dt_gilb, N);
  gpu_sync();
}

#ifdef __cplusplus
}
#endif
