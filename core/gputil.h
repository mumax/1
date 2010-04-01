/**
 * @file
 * This file provides some common functions for the GPU, like allocating arrays on it, FFT's.
 *
 * @author Arne Vansteenkiste
 */
#ifndef GPUTIL_H
#define GPUTIL_H

#include "tensor.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A complex-to-complex FFT plan on the GPU.
 * @see new_gpuc2cplan(), delete_gpuc2cplan(), gpuc2cplan_exec().
 */
typedef struct{  
  cufftHandle handle;
}gpuc2cplan;

/**
 * Creates a new 3D complex-to-complex FFT plan for the GPU.
 */
gpuc2cplan* new_gpuc2cplan(int N0,	///< size in x-direction
			   int N1,	///< size in y-direction
			   int N2	///< size in z-direction
			   );

/**
 * Forward FFT direction.
 * @see gpuc2cplan_exec()
 */
#define FORWARD	CUFFT_FORWARD


/**
 * Backward FFT direction.
 * @see gpuc2cplan_exec()
 */
#define INVERSE	CUFFT_INVERSE

/**
 * Executes the 3D complex-to-complex FFT plan in-place.
 */
void gpuc2cplan_exec(gpuc2cplan* plan,	///< the plan to be executed
		     float* data,	///< data to be transformed in-place
		     int direction	/// FORWARD or INVERSE
		     );

		     
/**
 * Frees the FFT plan
 * @todo not fully implemented
 */
void delete_gpuc2cplan(gpuc2cplan* plan	///< the plan to be deleted
		      );

/**
 * In many cases, CUDA arrays are made "too big" by appending zero's until the size
 * is a multiple of 512 (or in general, the number of threads per block). This function
 * takes a size as parameter and returns the smallest multiple of 512, larger than
 * or equal to size.
 * @see new_gpu_array()
 */
int gpu_len(int size	///< minimum size of the array
	    );

/**
 * Allocates an array of floats on the GPU and asserts the size is a multiple of 512.
 * @see gpu_len(), new_ram_array()
 */
float* new_gpu_array(int size	///< size of the array
		    );

/**
 * Allocates an array of floats in the main RAM.
 * @see gpu_len(), new_gpu_array()
 */		    
float* new_ram_array(int size	///< size of the array
		    );

/**
 * Copies floats from the main RAM to the GPU.
 * @see memcpy_from_gpu(), memcpy_gpu_to_gpu()
 */
void memcpy_to_gpu(float* source,	///< source data pointer in the RAM
		   float* dest,		///< destination data pointer on the GPU
		   int nElements	///< number of floats (not bytes) to be copied
		   );

/**
 * Copies floats from GPU to the main RAM.
 * @see memcpy_to_gpu(), memcpy_gpu_to_gpu()
 */
void memcpy_from_gpu(float* source,	///< source data pointer on the GPU
		     float* dest,	///< destination data pointer in the RAM
		     int nElements	///< number of floats (not bytes) to be copied
		     );

/**
 * Copies floats from GPU to GPU.
 * @see memcpy_to_gpu(), memcpy_from_gpu()
 */
void memcpy_gpu_to_gpu(float* source,	///< source data pointer on the GPU
		       float* dest, 	///< destination data pointer on the GPU
		       int nElements	///< number of floats (not bytes) to be copied
		      );
		 
/**
 * Set a range of floats on the GPU to zero.
 */
void gpu_zero(float* data,	///< data pointer on the GPU
	      int nElements	///< number of floats (not bytes) to be zeroed
	      );

/**
 * This function should be wrapped around cuda functions to check for a non-zero error status.
 * It will print an error message and abort when neccesary.
 */
void gpu_safe(int status	///< CUDA return status
	      );

/**
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 */
void gpu_checkconf(dim3 gridsize, ///< 3D size of the thread grid
		   dim3 blocksize ///< 3D size of the trhead blocks on the grid
		   );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 */	   
void gpu_checkconf_int(int gridsize, ///< 1D size of the thread grid
		       int blocksize ///< 1D size of the trhead blocks on the grid
		       );

#ifdef __cplusplus
}
#endif
#endif