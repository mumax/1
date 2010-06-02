/**
 * @file
 * This file provides some common functions for the GPU, like allocating arrays on it...
 *
 * @todo use CudaGetDeviceProperties to obtain the maximum number of threads per block, etc...
 * @todo Strided Arrays
 * @todo Smart zero-padded FFT's: try strided and transposed
 * @todo choose between in-place and out-of-place FFT's for best performance or best memory efficiency
 * 
 * @author Arne Vansteenkiste
 */
#ifndef GPUTIL_H
#define GPUTIL_H

#include "tensor.h"
#include <cufft.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Allocates an array of floats on the GPU and asserts the size is a multiple of 512.
 * @see new_ram_array()
 */
float* new_gpu_array(int size	///< size of the array
		    );
/**
 * Creates a new tensor whose data is allocated on the GPU. (rank and size are stored in the host RAM)
 */
// tensor* new_gpu_tensor(int rank, ...);

/**
 * Allocates an array of floats in the main RAM.
 * @see new_gpu_array()
 */		    
float* new_ram_array(int size	///< size of the array
		    );

/**
 * Returns the optimal array stride (in number of floats):
 * the second dimension of a 2D array should be a multiple of the stride.
 * This number is usually 64 but could depend on the hardware.
 *
 * E.g.: it is better to use a  3 x 64 array than a 64 x 3.
 * 
 * This seems to generalize to higher dimensions: at least the last
 * dimension should be a multiple of the stride. E.g.:
 * Standard problem 4 ran about 4x faster when using a (3x) 1 x 32 x 128 geometry
 * instead of (3x) 128 x 32 x 1 !
 *
 * @todo use cudaGetDeviceProperties for this?
 */
int gpu_stride_float();


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
 * @code
 * gpu_safe( cudaMalloc(...) );
 * @endcode
 */
void gpu_safe(int status	///< CUDA return status
	      );

/**
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @todo: use device properties
 */
void gpu_checkconf(dim3 gridsize, ///< 3D size of the thread grid
		   dim3 blocksize ///< 3D size of the trhead blocks on the grid
		   );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @todo: use device properties
 */	   
void gpu_checkconf_int(int gridsize, ///< 1D size of the thread grid
		       int blocksize ///< 1D size of the trhead blocks on the grid
		       );
		       
/**
 * Makes a 3D thread configuration suited for a data array of size N0 x N1 x N2.
 * The returned configuration will:
 *  - span the entire N0 x N1 x N2 array
 *  - have the largest valid block size that fits in the N0 x N1 x N2 array
 * 
 */
void make3dconf(int N0, 	///< size of 3D array to span
		int N1, 	///< size of 3D array to span
		int N2, 	///< size of 3D array to span
		dim3* gridSize, ///< grid size is returned here
		dim3* blockSize ///< block size is returned here
		);

/**
 * Prints the properties of the used GPU
 */
void print_device_properties(FILE* out	///< stream to print to
			     );

#ifdef __cplusplus
}
#endif
#endif