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
#ifndef GPU_FFT_H
#define GPU_FFT_H

#include "tensor.h"
#include "gpukern.h"
#include <cufft.h>
// #include <gpu_transpose2.h>

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
  
  cufftHandle fwPlanZ;     ///< 1D real-to-complex plan for Z-direction
  cufftHandle invPlanZ;    ///< 1D complex-to-real plan for Z-direction
  cufftHandle planY;       ///< 1D complex-to-complex plan for Y-direction, forward or inverse
  cufftHandle planX;       ///< 1D complex-to-complex plan for X-direction, forward or inverse
  
  float* transp;           ///< buffer for out-of-place transposing
  
}gpuFFT3dPlan;



/**
 * Creates a new FFT plan for transforming real 2D or 3D data. 
 * Zero-padding in each dimension is optional, and rows with only zero's are not transformed.
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, 
                                      int* paddedSize
                                      );

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, tensor* input, tensor* output);
                                      
/**
 * Forward FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * Routine is called 'unsafe' since the input is not checked for compatibility: input data are float arrays.
 */
void gpuFFT3dPlan_forward_unsafe(gpuFFT3dPlan* plan, 
                                 float* input, 
                                 float* output
                                 );                          

                                 
                                 
/**
 * Forward FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * The input is checked for compatibility: input data are tensors.
 */
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, 
                          tensor* input, 
                          tensor* output);                            
                                 
                                 
/**
 * Inverse FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * Routine is called 'unsafe' since the input is not checked for compatibility: input data are float arrays.
 */
void gpuFFT3dPlan_inverse_unsafe(gpuFFT3dPlan* plan, 
                                 float* input, 
                                 float* output
                                 );                               
                                 

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
 * @internal Does padding and unpadding, not necessarily by a factor 2
 **/
__global__ void _gpu_copy_pad(float* source,        ///< source data
                              float* dest,          ///< destination data
                              int S1,               ///< source size Y
                              int S2,               ///< source size Z
                              int D1,               ///< destination size Y
                              int D2                ///< destination size Z
                              );
                              
/**
 * @internal
 * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_pad(float* source,         ///< input: unpadded source as contiguous float array
                     float* dest,           ///< output: padded destination as contiguous float array
                     int *unpad_size4d,     ///< size of the corresponding unpadded tensor 
                     int *pad_size4d        ///< size of the corresponding padded tensor
                     );

/**
 * @internal
 * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_unpad(float* source,        ///< input: padded source as contiguous float array
                       float* dest,          ///< output: unpadded destination as contiguous float array
                       int *pad_size4d,      ///< size of the corresponding padded tensor
                       int *unpad_size4d     ///< size of the corresponding unpadded tensor 
                       ); 

                               
                               
                               
#ifdef __cplusplus
}
#endif
#endif