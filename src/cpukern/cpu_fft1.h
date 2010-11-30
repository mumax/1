/**
 * @file
 *
 * @todo cleanup
 *
 * @author Ben Van de Wiele
 */

#ifndef cpu_fft1_h
#define cpu_fft1_h
#include "fftw3.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A real-to-complex FFT plan on the CPU.
 */
typedef struct{
  int* size;                    ///< logical size of the (real) input data
  int* paddedSize;              ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".

  fftwf_plan FFT_mt_FW_dim1;
  fftwf_plan FFT_mt_FW_dim2;
  fftwf_plan FFT_mt_FW_dim3;
  fftwf_plan FFT_mt_BW_dim1;
  fftwf_plan FFT_mt_BW_dim2;
  fftwf_plan FFT_mt_BW_dim3;
  
}cpuFFT3dPlan;

/**
 * Initializes an FFT plan (forward and inverse) with possible zero padding
 */
cpuFFT3dPlan* new_cpuFFT3dPlan_padded(int* size,        ///>logical size of the (real) input data
                                      int* paddedSize   ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
                                      );

/**
 *Frees a cpuFFT3dPlan
 */
void delete_cpuFFT3dPlan(cpuFFT3dPlan* plan         ///<plan to be freed
                        );                                      

/**
 * @internal
 * Forward (real-to-complex) transform.
 * Sizes are not checked.
 * @see cpuFFT3dPlan_forward()
 */
void cpuFFT3dPlan_forward(cpuFFT3dPlan* plan,       ///< the plan to be executed
                         float* input,      ///< input data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                         float* output      ///< output data, may be equal to input for in-place transforms.
                          );

/**
 * @internal
 * Backward (complex-to-real) transform.
 * Sizes are not checked.
 * @see cpuFFT3dPlan_inverse()
 */
void cpuFFT3dPlan_inverse(cpuFFT3dPlan* plan,       ///< the plan to be executed
                         float* input,            ///< input data, may be equal to output for in-place transforms.
                         float* output            ///< output data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                          );
                                 
///@todo access to X,Y,Z transforms for multi GPU / MPI implementation


#ifdef __cplusplus
}
#endif
#endif
