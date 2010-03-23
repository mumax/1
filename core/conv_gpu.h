#ifndef CONV_GPU_H
#define CONV_GPU_H

#include "tensor.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" { // allow inclusion in C++ code
#endif


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
  
  /** Physical size (number of floats) of padded, complex data. Currently: size[0], size[1], 2*size[2], can become size[0], size[1], (size[2] + 2). */
  int paddedComplexSize[3];
  /** Total number of real elements in padded complex data (currently = 2 * size[0] * size[1] * size[2]).*/
  int paddedComplexN;
  
  /** Transformed magnetization component m_i, re-used for m_0, m_1, m_2 (allocating three separate buffers would be a waste of space). This space is zero-padded to be twice the size of the input data in all directions. Plus, the last dimension is even 2 elements larger to allow an in-place FFT. */
  tensor* ft_m_i; //rank 3
  
   /** Transformed total demag field due to all the magnetization components i: h[j] = sum_i h_i[j] */
   tensor* ft_h; //rank 4
   
}convplan;


/** Makes a new convplan with given logical size of the input data and a convolution kernel. */
convplan* new_convplan(int* size, tensor* kernel);

void delete_convplan(convplan* plan);

void conv_execute(convplan* plan, tensor* source, tensor* dest);


#ifdef __cplusplus
}
#endif

#endif