#ifndef GPUCONV1_H
#define GPUCONV1_H

#include "tensor.h"
#include "gputil.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 */

typedef struct{
  
  int* size;
  int N;
  
  int* paddedSize;
  int paddedN;
  
  int* paddedComplexSize;
  int paddedComplexN;
  
  int len_m;
  float** m_comp;
  int len_m_comp;
  float* ft_m_i;
  int len_ft_m_i;
  
  float*** ft_kernel;	// first two indices reside on the RAM, last index on the GPU
  int len_ft_kernel;
  int len_ft_kernel_ij;
  int len_kernel_ij;
  
  float* h;
  int len_h;
  float** h_comp;
  int len_h_comp;
  float* ft_h;
  int len_ft_h;
  float** ft_h_comp;
  int len_ft_h_comp;
  
  gpuc2cplan* fftplan;
  
}gpuconv1;


gpuconv1* new_gpuconv1(int N0, int N1, int N2, tensor* kernel);

void gpuconv1_updateh(gpuconv1* sim, float* m, float* h);

void gpuconv1_loadkernel(gpuconv1* sim, tensor* kernel);

void memcpy_r2c(float* source, float* dest, int nReal);

void gpu_copy_pad_r2c(float* source, float* dest, int N0, int N1, int N2);
void gpu_copy_unpad_c2r(float* source, float* dest, int N0, int N1, int N2);
void gpu_kernel_mul(float* ft_m_i, float* ft_kernel_comp_ij, float* ft_h_comp_j, int nRealNumbers);


#ifdef __cplusplus
}
#endif
#endif