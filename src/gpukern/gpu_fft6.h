/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 *
 * This file collects all FFT routines for possibly zeropadded matrices with 2D or 3D dimensions.
 * The following is taken into account:
 *    - real-to-complex FFTs
 *    - No FFTs on rows which contain only zeros
 *    - The CUCA memory access is aligned
 *    - 2D FFTs on data padded in X- and/or Y-direction is performed in-place
 *    
 * @todo restrict the required extra memory for 3D to the minimum
 * @todo concurrent execution?
 
 * @author Ben Van de Wiele
 * @author Arne Vansteenkiste
 */
#ifndef GPU_FFT6_H
#define GPU_FFT6_H

#include "../macros.h"
#include "gpu_mem.h"
#include <cufft.h>
#include "gpu_transpose.h"
#include "gpu_transpose2.h"
#include "gpu_safe.h"
#include "gpu_conf.h"
#include "gpu_zeropad.h"
#include "gpu_fftbig.h"

#ifdef __cplusplus
extern "C" {
#endif



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
  
}gpuFFT3dPlan;



/**
 * Creates a new FFT plan for transforming real 2D or 3D data. 
 * Zero-padding in each dimension is optional, and rows with only zero's are not transformed.
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, 
                                      int* paddedSize
                                      );

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan,
                          float* input, 
                          float* output
                          );

void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan,
                          float* input,
                          float* output
                          );
                              
                              
/**
 * returns the normalization factor of a given fft plan.
 */
int gpuFFT3dPlan_normalization(gpuFFT3dPlan* plan);

/**
 * In this routine, the input data 'data' is Fourier transformed in the Z-direction and stored contiguously starting from the 
 * seconds half of the data array and will be transposed towards the first part of the matrix.  Then it is ready for the transform 
 * in the Y-direction.  Zero-padding in X- and/or Y-direction is assumed.
 */
void yz_transpose_in_place_fw(float *data, 
                              int *size, 
                              int *pSSize
                              );

/**
 * In this routine, all input FFT transformed data is stored non-contiguously starting from the first half of the  data array and 
 * will be transposed towards the second part of the matrix.  Then it is ready for the transform in the Z-direction.
 * Zero-padding in X- and/or Y-direction is assumed.
 */
void yz_transpose_in_place_inv(float *data, 
                               int *size, 
                               int *pSSize
                               );
                               

                               
// functions for copying to and from padded matrix ****************************************************
/**
 * @internal
 * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_pad(float* source,         ///< input: unpadded source as contiguous float array
                     float* dest,           ///< output: padded destination as contiguous float array
                     int *unpad_size,       ///< size of the corresponding unpadded tensor 
                     int *pad_size          ///< size of the corresponding padded tensor
                     );

/**
 * @internal
 * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_unpad(float* source,        ///< input: padded source as contiguous float array
                       float* dest,          ///< output: unpadded destination as contiguous float array
                       int *pad_size,        ///< size of the corresponding padded tensor
                       int *unpad_size       ///< size of the corresponding unpadded tensor 
                       ); 



/**
 * @internal
 * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_pad2(float* source,         ///< input: unpadded source as contiguous float array
                      float* dest,           ///< output: padded destination as contiguous float array
                      int *unpad_size,       ///< size of the corresponding unpadded tensor 
                      int *pad_size          ///< size of the corresponding padded tensor
                      );

/**
 * @internal
 * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_unpad2(float* source,        ///< input: padded source as contiguous float array
                        float* dest,          ///< output: unpadded destination as contiguous float array
                        int *pad_size,        ///< size of the corresponding padded tensor
                        int *unpad_size       ///< size of the corresponding unpadded tensor 
                        ); 
             
/**
 * @internal
 * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_pad3(float* source,         ///< input: unpadded source as contiguous float array
                      float* dest,           ///< output: padded destination as contiguous float array
                      int *unpad_size,       ///< size of the corresponding unpadded tensor 
                      int *pad_size          ///< size of the corresponding padded tensor
                      );

/**
 * @internal
 * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_unpad3(float* source,        ///< input: padded source as contiguous float array
                        float* dest,          ///< output: unpadded destination as contiguous float array
                        int *pad_size,        ///< size of the corresponding padded tensor
                        int *unpad_size       ///< size of the corresponding unpadded tensor 
                        ); 

void delete_gpuFFT3dPlan(gpuFFT3dPlan* plan);
                               
#ifdef __cplusplus
}
#endif
#endif