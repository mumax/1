/**
 * @file 
 * Implements the cpu-multithreaded version of the fast Fourier transforms without transforms on zero arrays
 *
 * @author Ben Van de Wiele
 */

#ifndef CPU_FFT2_H
#define CPU_FFT2_H

#include "fftw3.h"
#include "thread_functions.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct{
  int *size;                    ///< logical size of the (real) input data
  int *paddedSize;              ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
  thread_data *T_data;          ///< threaddata

  int *N_FFT_dimx;
  int *N_FFT_dimy;
  int *N_FFT_dimyz;

  fftwf_plan *FFT_FW_dim1;
  fftwf_plan FFT_FW_dim2;
  fftwf_plan FFT_FW_dim3;
  fftwf_plan *FFT_BW_dim1;
  fftwf_plan FFT_BW_dim2;
  fftwf_plan FFT_BW_dim3;
  
}cpuFFT3dPlan;




/**
 * Initializes an FFT plan (forward and inverse) with possible zero padding, multithreaded
 */
cpuFFT3dPlan* new_cpuFFT3dPlan (int* size,           ///> logical size of the (real) input data
                                int* paddedSize      ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
                               );

/**
 *Frees a cpuFFT3dPlan
 */
void delete_cpuFFT3dPlan(cpuFFT3dPlan* plan           ///<plan to be freed
                         );                                      

/**
 * @internal
 * Forward (real-to-complex) transform, multithreaded.
 * Sizes are not checked.
 * @see cpuFFT3dPlan_forward()
 */
void cpuFFT3dPlan_forward(cpuFFT3dPlan* plan,  ///< the plan to be executed
                          float* input,           ///< input data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                          float* output           ///< output data, may be equal to input for in-place transforms.
                          );

/**
 * @internal
 * Inverse (complex-to-real) transform, multithreaded.
 * Sizes are not checked.
 * @see cpuFFT3dPlan_inverse()
 */
void cpuFFT3dPlan_inverse(cpuFFT3dPlan* plan,  ///< the plan to be executed
                          float* input,           ///< input data, may be equal to output for in-place transforms.
                          float* output           ///< output data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                          );


#ifdef __cplusplus
}
#endif
#endif
