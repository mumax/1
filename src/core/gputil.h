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

//_________________________________________________________________________________________ allocation

/**
 * Allocates an array of floats on the GPU and asserts the size is a multiple of 512.
 * @see new_ram_array()
 */
float* new_gpu_array(int size	///< size of the array
                    );
                    
/**
 * Creates a new tensor whose data is allocated on the GPU. (rank and size are stored in the host RAM)
 * @todo delete_gputensor()
 */
 tensor* new_gputensor(int rank, int* size);

/**
 * Allocates an array of floats in the main RAM.
 * @see new_gpu_array()
 */		    
float* new_ram_array(int size       ///< size of the array
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
 * @see gpu_pad_to_stride()
 */
int gpu_stride_float();

/**
 * This function takes an array size (in number of floats) and returns an
 * array size -usually larger- that can store the original array and fits
 * the GPU stride.
 * Example (for a stride of 64 floats -- 256 bytes):
 @code
 *  1 -> 64
 *  2 -> 64
 * ...
 * 63 -> 64
 * 64 -> 64
 * 65 -> 128
 * ...
 @endcode
 */
int gpu_pad_to_stride(int nFloats);

/**
 * @internal For debugging, it is handy to use a smaller-than-optimal stride;
 * this prevents small test data to be padded to huge proportions. To reset
 * to the intrinsic machine stride, set the value to -1.
 */
void gpu_override_stride(int nFloats    ///< The stride (in number of floats) to use instead of the real, GPU-dependent stride.
                        );

//______________________________________________________________________________________ copy

/**
 * Copies floats from the main RAM to the GPU.
 * @see memcpy_from_gpu(), memcpy_on_gpu()
 */
void memcpy_to_gpu(float* source,	///< source data pointer in the RAM
		   float* dest,		///< destination data pointer on the GPU
		   int nElements	///< number of floats (not bytes) to be copied
		   );

/**
 * Copies floats from GPU to the main RAM.
 * @see memcpy_to_gpu(), memcpy_on_gpu()
 */
void memcpy_from_gpu(float* source,	///< source data pointer on the GPU
		     float* dest,	///< destination data pointer in the RAM
		     int nElements	///< number of floats (not bytes) to be copied
		     );

/**
 * Copies floats from GPU to GPU.
 * @see memcpy_to_gpu(), memcpy_from_gpu()
 */
void memcpy_on_gpu(float* source,	///< source data pointer on the GPU
                       float* dest, 	///< destination data pointer on the GPU
                       int nElements	///< number of floats (not bytes) to be copied
                       );

/// @internal Reads one float from a GPU array, not extremely efficient.
float gpu_array_get(float* dataptr, int index);

/// @internal Writes one float to a GPU array, not extremely efficient.
void gpu_array_set(float* dataptr, int index, float value);


/**
 * Copies the source tensor (in RAM) to the the destination tensor (on the GPU).
 * They should have equal sizes.
 * @see tensor_copy_from_gpu(), tensor_copy_on_gpu()
 */
void tensor_copy_to_gpu(tensor* source, tensor* dest);

/**
 * Copies the source tensor (on the GPU) to the the destination tensor (in RAM).
 * They should have equal sizes.
 * @see tensor_copy_to_gpu(), tensor_copy_on_gpu()
 */
void tensor_copy_from_gpu(tensor* source, tensor* dest);

/**
 * Copies the source tensor to the the destination tensor (both on the GPU).
 * They should have equal sizes.
 * @see tensor_copy_to_gpu(), tensor_copy_from_gpu()
 */
void tensor_copy_on_gpu(tensor* source, tensor* dest);


//______________________________________________________________________________________ util

/**
 * Set a range of floats on the GPU to zero.
 */
// void gpu_zero(float* data,	///< data pointer on the GPU
//               int nElements	///< number of floats (not bytes) to be zeroed
//               );

/**
 * Sets all the tensor's elements to zero. The tensor should be allocated on the GPU.
 */
void gpu_zero_tensor(tensor* t);

/**
 * This function should be wrapped around cuda functions to check for a non-zero error status.
 * It will print an error message and abort when neccesary.
 * @code
 * gpu_safe( cudaMalloc(...) );
 * @endcode
 */
// void gpu_safe(int status	///< CUDA return status
// 	      );

/**
 * @internal
 * Debug function for printing gpu tensors without first having
 * to copy them to host memory manually. 
 */
void format_gputensor(tensor* t, FILE* out);


/**
 * @internal
 * Checks if the data resides on the host by copying one float from host to host.
 * A segmentation fault is thrown when the data resides on the GPU device.
 */
void assertHost(float* pointer);


/**
 * @internal
 * Checks if the data resides on the GPU device by copying one float from device to device.
 * A segmentation fault is thrown when the data resides on the host.
 */
void assertDevice(float* pointer);


//______________________________________________________________________________________ check conf

/**
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @deprecated use check3dconf(), which uses the actual device properties
 */
// void gpu_checkconf(dim3 gridsize, ///< 3D size of the thread grid
// 		   dim3 blocksize ///< 3D size of the trhead blocks on the grid
// 		   );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * @deprecated use check1dconf(), which uses the actual device properties
 */	   
// void gpu_checkconf_int(int gridsize, ///< 1D size of the thread grid
// 		       int blocksize ///< 1D size of the trhead blocks on the grid
// 		       );
		       
/**
 * Checks if the CUDA 3D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */
// void check3dconf(dim3 gridsize, ///< 3D size of the thread grid
// 		   dim3 blocksize ///< 3D size of the trhead blocks on the grid
// 		   );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid. 
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */	   
// void check1dconf(int gridsize, ///< 1D size of the thread grid
// 		       int blocksize ///< 1D size of the trhead blocks on the grid
// 		       );
		       

//______________________________________________________________________________________ make conf

/**
 * Makes a 3D thread configuration suited for a float array of size N0 x N1 x N2.
 * The returned configuration will:
 *  - span the entire N0 x N1 x N2 array
 *  - have the largest valid block size that fits in the N0 x N1 x N2 array
 *  - be valid
 *
 * @todo works only up to N2 = 512 
 * @see make1dconf()
 *
 * Example:
 * @code
  dim3 gridSize, blockSize;
  make3dconf(N0, N1, N2, &gridSize, &blockSize);
  mykernel<<<gridSize, blockSize>>>(arrrrgh);
 * @endcode
 */
// void make3dconf(int N0, 	///< size of 3D array to span
// 		int N1, 	///< size of 3D array to span
// 		int N2, 	///< size of 3D array to span
// 		dim3* gridSize, ///< grid size is returned here
// 		dim3* blockSize ///< block size is returned here
// 		);
		
/**
 * Makes a 1D thread configuration suited for a float array of size N
 * The returned configuration will:
 *  - span the entire array
 *  - have the largest valid block size that fits in the  array
 *  - be valid
 *
 * @see make3dconf()
 *
 * Example:
 * @code
 * int gridSize, blockSize;
 * make1dconf(arraySize, &gridSize, &blockSize);
 * mykernel<<<gridSize, blockSize>>>(arrrrgh);
 * @endcode
 */
// void make1dconf(int N,          ///< size of array to span (number of floats)
//                 int* gridSize,  ///< grid size is returned here
//                 int* blockSize  ///< block size is returned here
//                 );


/**
 * @internal
 * Returns a cudaDeviceProp struct that contains the properties of the
 * used GPU. When there are multiple GPUs present, the active one, used
 * by this thread, is considered.
 *
 * @warning One global cudaDeviceProp* is stored. The first time this
 * function is called, it gets initialized. All subsequent calls return
 * this cached cudaDeviceProp*. Consequently, the returned pointer
 * must not be freed!
 *
 * The struct looks like this:
 * @code
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    size_t totalConstMem;
    int major;
    int minor;
    int clockRate;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
 * @endcode
 *
 * @note I currently return the cudaDeviceProp* as a void*. 
 * In this way, none of the core functions expose cuda stuff
 * directly. This makes it easier to link them with external
 * code (Go, in my case). Arne.
 */
void* gpu_getproperties(void);

/**
 * Prints the properties of the used GPU
 */
void print_device_properties(FILE* out	///< stream to print to
);

void print_device_properties_stdout();

#ifdef __cplusplus
}
#endif
#endif