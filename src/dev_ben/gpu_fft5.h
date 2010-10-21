/**
 * @file
 *
 * This file collects all FFT routines for possibly zeropadded matrices with 2D or 3D dimensions.
 * The following is taken into account:
 *    - real-to-complex FFTs
 *    - No FFTs on rows which contain only zeros
 *    - The CUDA memory access is aligned
 *    - 2D FFTs on data padded in X- and/or Y-direction is performed in-place
 *    - The FFTs are split in several batches to fit the maximum elements treated in CUFFT
 *    
 * @todo restrict the required extra memory for 3D to the minimum
 * @todo concurrent execution?
 
 * @author Ben Van de Wiele
 * @author Arne Vansteenkiste
 */
#ifndef GPU_FFT5_H
#define GPU_FFT5_H


#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_fftbig.h"
#include "gpu_fft4.h"


/**
* A real-to-complex FFT plan on the GPU.
*/
typedef struct{
  int* size;               ///< logical size of the (real) input data
  int N;                   ///< total number of floats in size
  
  int* paddedSize;         ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
  int paddedN;             ///< total number of floats in paddedSize
  
  int* paddedStorageSize;  ///< A real-to-complex FFT requires padding with one complex number in the last dimension. However, is this would result in misalgned memory, we pad with (typically) 64 floats
  int paddedStorageN;      ///< total number of floats in paddedStorageSize

  bigfft *fwPlanZ;         ///< 1D real-to-complex plan for Z-direction
  bigfft *invPlanZ;        ///< 1D complex-to-real plan for Z-direction
  bigfft *planY;           ///< 1D complex-to-complex plan for Y-direction, forward or inverse
  bigfft *planX;           ///< 1D complex-to-complex plan for X-direction, forward or inverse
  
  float* transp;           ///< buffer for out-of-place transposing
  
}gpuFFT3dPlan_big;



/**
 * Creates a new FFT plan for transforming real 2D or 3D data. 
 * Zero-padding in each dimension is optional, and rows with only zero's are not transformed.
 */
gpuFFT3dPlan_big* new_gpuFFT3dPlan_padded_big(int* size, 
                                              int* paddedSize
                                              );

void gpuFFT3dPlan_forward_big(gpuFFT3dPlan_big* plan,
                              float* input, 
                              float* output
                              );

void gpuFFT3dPlan_inverse_big(gpuFFT3dPlan_big* plan,
                              float* input,
                              float* output
                              );
                              
#ifdef __cplusplus
}
#endif
#endif