#ifndef CONV_GPU_H
#define CONV_GPU_H

#include "tensor.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" { // allow inclusion in C++ code
#endif


/**
 * A plan for 3D vector convolutions, analogous to an FFTW plan.
 */
typedef struct{
  int* size;	// the logical size of the input data (array of length 3)
}convplan;

/** Makes a new convplan with given logical size of the input data and a convolution kernel. */
convplan* new_convplan(int* size, tensor* kernel);


#ifdef __cplusplus
}
#endif

#endif