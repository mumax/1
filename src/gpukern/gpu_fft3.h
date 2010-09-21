/**
 * @file
 *
<<<<<<< HEAD
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
=======
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_fft3_h
#define gpu_fft3_h
>>>>>>> arne

#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
<<<<<<< HEAD
* A real-to-complex FFT plan on the GPU.
*/
typedef struct{
  int* size;               ///< logical size of the (real) input data
  int N;                   ///< total number of floats in size
  
  int* paddedSize;         ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
  int paddedN;             ///< total number of floats in paddedSize
  
  int* paddedStorageSize;  ///< A real-to-complex FFT requires padding with one complex number in the last dimension. However, is this would result in misalgned memory, we pad with (typically) 64 floats
  int paddedStorageN;      ///< total number of floats in paddedStorageSize
=======
 * A real-to-complex FFT plan on the GPU.
 */
typedef struct{
  int* size;               ///< logical size of the (real, unpadded) input data
  int  dataN;              ///< total number of floats in dataSize

  int* paddedSize;         ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "dataSize".
  int  paddedN;            ///< total number of floats in paddedSize

//   int* paddedComplexSize;  ///< physical size of the (complex, padded) output data in half-complex format
//   int  paddedComplexN;
>>>>>>> arne
  
  cufftHandle fwPlanZ;     ///< 1D real-to-complex plan for Z-direction
  cufftHandle invPlanZ;    ///< 1D complex-to-real plan for Z-direction
  cufftHandle planY;       ///< 1D complex-to-complex plan for Y-direction, forward or inverse
  cufftHandle planX;       ///< 1D complex-to-complex plan for X-direction, forward or inverse
<<<<<<< HEAD
  
  float* transp;           ///< buffer for out-of-place transposing @todo: CAN WE REMOVE THIS ONE?
  
=======

  float* buffer1;          ///< Buffer for zero-padding in Z
  float* buffer2;          ///< Buffer for result of out-of-place FFT_z
  float* buffer2t;         ///< buffer2 transposed
  float* buffer3;          ///< Buffer for zero-padding in Y and in-place transform
  float* buffer3t;         ///< buffer3 transposed

>>>>>>> arne
}gpuFFT3dPlan;



/**
<<<<<<< HEAD
 * Creates a new FFT plan for transforming real 2D or 3D data. 
 * Zero-padding in each dimension is optional, and rows with only zero's are not transformed.
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, 
                                      int* paddedSize
                                      );

/**
 * Forward FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * The input is checked for compatibility: input data are tensors.
 */
// void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, 
//                           tensor* input, 
//                           tensor* output
//                           );
                          
                          
/**
 * Forward FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * Routine is called 'unsafe' since the input is not checked for compatibility: input data are float arrays.
 */
void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan,
                                 float* input, 
                                 float* output
                                 );                          

                                 
                                 
/**
 * Forward FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * The input is checked for compatibility: input data are tensors.
 */
/*void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, 
                          tensor* input, 
                          tensor* output); */                           
                                 
                                 
/**
 * Inverse FFT of real possibly zero-padded data with 2D or 3D dimensions. FFTs on rows containing only zeros are not performed.
 * Routine is called 'unsafe' since the input is not checked for compatibility: input data are float arrays.
 */
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan,
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
#ifdef __cplusplus
}
#endif
#endif
=======
 * Creates a new real-to-complex 3D FFT plan with efficient handling of padding zeros.
 * If paddedsize is larger than size, then the additional space is filled with zeros,
 * but they are efficiently handled during the transform.
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size,       ///< size of real input data (3D)
                                      int* paddedsize  ///< size of the padded data (3D). Should be at least the size of the input data. If the kernel is larger, the input data is assumed to be padded with zero's which are efficiently handled by the FFT
                                      );

/**
 * @internal
 * Forward (real-to-complex) transform.
 * Sizes are not checked.
 * @see gpuFFT3dPlan_inverse()
 */
void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan,      ///< the plan to be executed
                          float* input,            ///< input data. Size = dataSize. Real
                          float* output            ///< output data. Size = paddedComplexSize. Half-complex format
                          );

/**
 * @internal
 * Backward (complex-to-real) transform.
 * Sizes are not checked.
 * @see gpuFFT3dPlan_forward()
 */
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan,       ///< the plan to be executed
                                 float* input,      ///< input data, Size = paddedComplexSize. Half-complex format
                                 float* output      ///< output data, Size = dataSize. Real
                                 );
                                 
///@todo access to X,Y,Z transforms for multi GPU / MPI implementation

#ifdef __cplusplus
}
#endif
#endif
>>>>>>> arne
