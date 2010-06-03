/**
 * @file
 * Initialization, execution and deleting of FFTs used during the computation of the micromagnetic kernel.
 * The following is taken into account:
 *   - real-to-complex FFTs
 *   - No FFTs on the zeros in the padded magnetization buffers
 *   - The CUDA memory access is alligned
 *
 * @todo In place transpose routines
 *
 * @see gpuconv1, new_gpuconv2, gpuconv2_exec
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef GPUFFT_H
#define GPUFFT_H

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
  int* size;		          ///< logical size of the (real) input data
  int N;		              ///< total number of floats in size

  int* paddedSize;	      ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
  int paddedN;		        ///< total number of floats in paddedSize

  int* paddedStorageSize;	///< A real-to-complex FFT requires padding with one complex number in the last dimension. However, is this would result in misalgned memory, we pad with (typically) 64 floats
  int paddedStorageN;		  ///< total number of floats in paddedStorageSize

  cufftHandle fwPlanZ;	  ///< 1D real-to-complex plan for Z-direction
  cufftHandle invPlanZ;	  ///< 1D complex-to-real plan for Z-direction
  cufftHandle planY;	    ///< 1D complex-to-complex plan for Y-direction, forward or inverse
  cufftHandle planX;	    ///< 1D complex-to-complex plan for X-direction, forward or inverse

  float* transp;	        ///< buffer for out-of-place transposing

}gpu_plan3d_real_input;



/**
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @todo on compute capability < 2.0, the first step is done serially...
 */
gpu_plan3d_real_input* new_gpu_plan3d_real_input(int N0,            ///< size of real input data in x-direction
                                                 int N1,            ///< size of real input data  in y-direction
                                                 int N2,            ///< size of real input data  in z-direction
                                                 int* zero_pad      ///< 3 ints, should be 1 or 0, meaning zero-padding or no zero-padding in X,Y,Z respectively
                                                 );

/**
 * Executes in-place.
 */
void gpu_plan3d_real_input_forward(gpu_plan3d_real_input* plan, ///< the plan to be executed
                                   float* data                  ///< data to be transformed in-place, it size should be plan->paddedStorageSize
                                   );

/**
 * Executes in-place.
 */
void gpu_plan3d_real_input_inverse(gpu_plan3d_real_input* plan, ///< the plan to be executed
                                   float* data                  ///< data to be transformed in-place, it size should be plan->paddedStorageSize
                                   );

/**
 * @internal
 * Swaps the Y and Z components of a 3D array of complex numbers.
 * N0 x N1 x N2/2 complex numbers are stored as N0 x N1 x N2 interleaved real numbers.
 */
void gpu_transposeYZ_complex(float* source, ///< source data, size N0 x N1 x N2
                             float* dest,   ///< destination data, size N0 x N2 x N1
                             int N0,        ///< source size X
                             int N1,        ///< source size Z
                             int N2         ///< number of floats (!) in the Z-direction, thus 2x the number of complex numbers in Z.
                             );
/**
 * @internal
 * @see gpu_transposeYZ_complex()
 */
void gpu_transposeXZ_complex(float* source, 
			     float* dest, 
			     int N0, 
			     int N1, 
			     int N2
			     );

/**
 * Frees the FFT plan
 */
void delete_gpu_plan3d_real_input(gpu_plan3d_real_input* plan	///< the plan to be deleted
		      );

		    

#ifdef __cplusplus
}
#endif
#endif