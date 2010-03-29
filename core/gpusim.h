#ifndef CONV_H
#define CONV_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 */

typedef struct{
  
  int* size;
  int N;
  
  float* m;
  int len_m;
  
}gpusim;


gpusim* new_gpusim(int N0, int N1, int N2);

void gpusim_loadm(gpusim* sim, tensor* m);
void gpusim_dumpm(gpusim* sim, tensor* m);

void gpusim_loadkernel(gpusim* sim, tensor* kernel);

float* new_cuda_array(int size);

void memcpy_to_gpu(float* source, float* dest, int nElements);

void memcpy_from_gpu(float* source, float* dest, int nElements);


#ifdef __cplusplus
}
#endif
#endif