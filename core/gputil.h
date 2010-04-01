#ifndef GPUTIL_H
#define GPUTIL_H

#include "tensor.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 */

typedef struct{  
  cufftHandle handle;
}gpuc2cplan;



int gpu_len(int size);
float* new_gpu_array(int size);
float* new_ram_array(int size);

void memcpy_to_gpu(float* source, float* dest, int nElements);
void memcpy_from_gpu(float* source, float* dest, int nElements);
void memcpy_gpu_to_gpu(float* source, float* dest, int nElements);

void gpu_zero(float* data, int nElements);

gpuc2cplan* new_gpuc2cplan(int N0, int N1, int N2);
void gpuc2cplan_exec(gpuc2cplan* plan, float* data, int direction);
void delete_gpuc2cplan(gpuc2cplan* plan);

void gpu_safe(int status);

void gpu_checkconf(dim3 gridsize, dim3 blocksize);
void gpu_checkconf_int(int gridsize, int blocksize);

#ifdef __cplusplus
}
#endif
#endif