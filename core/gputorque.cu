#include "gputorque.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_deltaM(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float alpha, float dt_gilb){

  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);

  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  hx[i] = dt_gilb * (_mxHx + _mxmxHx * alpha);
  hy[i] = dt_gilb * (_mxHy + _mxmxHy * alpha);
  hz[i] = dt_gilb * (_mxHz + _mxmxHz * alpha);
  
}

void gpu_deltaM(float* m, float* h, float alpha, float dt_gilb, int N){

  int gridSize = -1, blockSize = -1;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  
  timer_start("torque");
  _gpu_deltaM<<<gridSize, blockSize>>>(mx, my, mz, hx, hy, hz, alpha, dt_gilb);
  cudaThreadSynchronize();
  timer_stop("torque");
}

#ifdef __cplusplus
}
#endif