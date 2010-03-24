#ifndef CONV_GPU_H
#define CONV_GPU_H

#include "tensor.h"
#include <stdio.h>
#include <cufft.h>

#ifdef __cplusplus
extern "C" { // allow inclusion in C++ code
#endif

typedef struct{
  cufftHandle handle;
  //tensor* device_data;
  float* device_buffer;
}cuda_c2c_plan;


/**
 * A convplan is a convolution plan for 3D vector convolutions, analogous to an FFTW plan.
 * It is set up once with a certain size of input data and kernel. Those can not change anymore.
 * The plan can then be called many times with the source (magnetization) and dest (field) arrays as parameters.
 */
typedef struct{
  /** Logical size of the input data. */
  int size[3];
  
  /** Total number of real elements in the input data (= size[0] * size[1] * size[2]). */
  int N;
  
  /** Size of the zero-padded data (2*size[0], 2*size[1], 2*size[2]) */
  int paddedSize[3];
  
   /** Physical size (number of floats) of padded, complex data. Currently: paddedSize[0], paddedSize[1], 2*paddedSize[2], can become paddedSize[0], paddedSize[1], (paddedSize[2] + 2). */
  int paddedComplexSize[3];
  
  /** Total number of real elements in padded complex data (currently = 2 * size[0] * size[1] * size[2]).*/
  int paddedComplexN;
  
  /** Transformed magnetization component m_i, re-used for m_0, m_1, m_2 (allocating three separate buffers would be a waste of space). This space is zero-padded to be twice the size of the input data in all directions. Plus, the last dimension is even 2 elements larger to allow an in-place FFT. */
  tensor* ft_m_i; //rank 3
  
   /** Transformed total demag field due to all the magnetization components i: h[j] = sum_i h_i[j] */
   tensor* ft_h; //rank 4
   tensor** ft_h_comp; // 3x rank 3
   
   /** Transformed Kernel */
   tensor* ft_kernel; // rank 5
   tensor*** ft_kernel_comp; // 3x3 x rank 3
   
   /** Plan for transforming size[0] * size[1] * size[2] complex numbers on the GPU. */
   cuda_c2c_plan* c2c_plan;
   
}convplan;


/** Makes a new convplan with given logical size of the input data and a convolution kernel (rank 5). */
convplan* new_convplan(int N0, int N1, int N2, float* kernel);

void delete_convplan(convplan* plan);

void conv_execute(convplan* plan, float* source, float* dest);

/** Initializes a 3D c2c FFT plan for the GPU. Size is the logical size of the input data (number of complex numbers). */
cuda_c2c_plan* gpu_init_c2c(int* size);

/** Executes a 3D c2c FFT plan created by gpu_init_c2c.*/
void gpu_exec_c2c(cuda_c2c_plan* plan, tensor* data, int direction);

/** Internal function that initializes the FT'ed kernel. */
void _init_kernel(convplan* plan, tensor* kernel);

#ifdef __cplusplus
}
#endif

#endif