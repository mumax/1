/**
 * @file
 * Like gpufft but works on tensors instead of float*'s
 *
 * Initialization, execution and deleting of FFTs used during the computation of the micromagnetic kernel.
 * The following is taken into account:
 *   - real-to-complex FFTs
 *   - No FFTs on the zeros in the padded magnetization buffers
 *   - The CUDA memory access is aligned
 *
 * @todo In place transpose routines
 * @todo Option for out-of place transforms
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef GPUFFT2_H
#define GPUFFT2_H

#include "tensor.h"
#include "gputil.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

//_________________________________________________________________________________________ FFT

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
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @todo on compute capability < 2.0, the first step is done serially...
 */
gpuFFT3dPlan* new_gpuFFT3dPlan(int* size,       ///< size of real input data (3D)
                               int* kernelsize  ///< size of the kernel (3D). Should be at least the size of the input data. If the kernel is larger, the input data is assumed to be padded with zero's which are efficiently handled by the FFT
                               );

/**
 * Forward (real-to-complex) transform.
 */
void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan,       ///< the plan to be executed
                          tensor* input,            ///< input data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                          tensor* output            ///< output data, may be equal to input for in-place transforms.
                          );

/**
 * Backward (complex-to-real) transform.
 */
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan,       ///< the plan to be executed
                          tensor* input,            ///< input data, may be equal to output for in-place transforms.
                          tensor* output            ///< output data, it's size should match the strided "half complex" format (=plan->paddedStorageSize)
                          );

/**
 * @internal
 * Swaps the Y and Z components of a 3D array of complex numbers.
 * N0 x N1 x N2/2 complex numbers are stored as N0 x N1 x N2 interleaved real numbers.
 */
void gpu_tensor_transposeYZ_complex(tensor* source, ///< source data, size N0 x N1 x (2*N2)
                                    tensor* dest   ///< destination data, size N0 x N2 x (2*N1)
                             );
/**
 * @internal
 * @see gpu_transposeYZ_complex()
 */
void gpu_tensor_transposeXZ_complex(tensor* source, ///< source data, size N0 x N1 x (2*N2)
                                    tensor* dest   ///< destination data, size N2 x N1 x (2*N0)
                             );

#ifdef __cplusplus
}
#endif
#endif