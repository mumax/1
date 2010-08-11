#include "cpu_torque.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_deltaM(float* m, float* h, float alpha, float dt_gilb, int N){

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  #pragma omp parallel for
  for(int i=0; i<N; i++){
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
}

#ifdef __cplusplus
}
#endif
