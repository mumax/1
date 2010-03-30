#ifndef CONV_H
#define CONV_H

#include "tensor.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 */

int threadsPerBlock = 512;
  
typedef struct{
  
  cufftHandle handle;
  
}gpusim_c2cplan;


typedef struct{
  
  int* size;
  int N;
  
  float* m;	// on GPU
  int len_m;
  float** m_comp;
  int len_m_comp;
  
  float* ft_m_i;
  int len_ft_m_i;
  
  float*** ft_kernel;	// first two indices reside on the RAM, last index on the GPU
  int len_ft_kernel;
  int len_ft_kernel_ij;
  int len_kernel_ij;
  
  float* ft_h_i;
  int len_ft_h_i;
  float* h;
  int len_h;
  float** h_comp;
  int len_h_comp;
  
}gpusim;


gpusim* new_gpusim(int N0, int N1, int N2, tensor* kernel);

void gpusim_updateh(gpusim* sim);

void gpusim_loadm(gpusim* sim, tensor* m);
void gpusim_storem(gpusim* sim, tensor* m);
void gpusim_loadkernel(gpusim* sim, tensor* kernel);

int gpu_len(int size);
float* new_gpu_array(int size);
float* new_ram_array(int size);

void memcpy_to_gpu(float* source, float* dest, int nElements);
void memcpy_from_gpu(float* source, float* dest, int nElements);
void memcpy_gpu_to_gpu(float* source, float* dest, int nElements);
void memcpy_r2c(float* source, float* dest, int nReal);
void gpu_copy_pad_r2c(float* source, float* dest, int N0, int N1, int N2);

void gpu_zero(float* data, int nElements);

gpusim_c2cplan* new_gpusim_c2cplan(int N0, int N1, int N2);
void gpusim_c2cplan_exec(gpusim_c2cplan* plan, float* data, int direction);
void delete_gpusim_c2cplan(gpusim_c2cplan* plan);

void gpusim_safe(int status);


#ifdef __cplusplus
}
#endif
#endif