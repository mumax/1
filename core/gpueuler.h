#ifndef GPUEULER_H
#define GPUEULER_H

#include "tensor.h"
#include "gpuconv1.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  
  int* size;
  int N;
  
  float* m;
  int len_m;
  
  float* h;
  int len_h;
  
  gpuconv1* convplan;
  
}gpueuler;

gpueuler* new_gpueuler(int N0, int N1, int N2, tensor* kernel);

void gpueuler_loadm(gpueuler* euler, tensor* m);
void gpueuler_storem(gpueuler* euler, tensor* m);

void gpueuler_step(gpueuler* solver, float dt);

#ifdef __cplusplus
}
#endif
#endif