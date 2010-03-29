#ifndef SOLVER_H
#define SOLVER_H

#ifdef __cplusplus
extern "C" { // allow inclusion in C++ code
#endif

#include "tensor.h"
#include "conv_gpu.h"

void torque(float mx, float my, float mz, float hx, float hy, float hz, float alpha, float gilbert, float* torque){

   // - m cross H
     float _mxHx = -my * hz + hy * mz;
     float _mxHy =  mx * hz - hx * mz;
     float _mxHz = -mx * hy + hx * my;

    // - m cross (m cross H)
     float _mxmxHx =  my * _mxHz - _mxHy * mz;
     float _mxmxHy = -mx * _mxHz + _mxHx * mz;
     float _mxmxHz =  mx * _mxHy - _mxHx * my;

    
    torque[X] = (_mxHx + _mxmxHx * alpha) * gilbert;
    torque[Y] = (_mxHy + _mxmxHy * alpha) * gilbert;
    torque[Z] = (_mxHz + _mxmxHz * alpha) * gilbert;
    
    return;
}

#ifdef __cplusplus
  }
#endif

#endif