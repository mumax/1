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

#include "gpu_safe.h"
#include "gpu_mem.h"
#include "tensor.h"
#include <cufft.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

//_________________________________________________________________________________________ allocation

                    
/**
 * Creates a new tensor whose data is allocated on the GPU. (rank and size are stored in the host RAM)
 * @todo delete_gputensor()
 */
 tensor* new_gputensor(int rank, int* size);


/**
 * Copies floats from GPU to GPU.
 * @see memcpy_to_gpu(), memcpy_from_gpu()
 */
void memcpy_gpu_to_gpu(float* source,	///< source data pointer on the GPU
                       float* dest, 	///< destination data pointer on the GPU
                       int nElements	///< number of floats (not bytes) to be copied
                       );

/**
 * Copies the source tensor (in RAM) to the the destination tensor (on the GPU).
 * They should have equal sizes.
 * @see tensor_copy_from_gpu(), tensor_copy_gpu_to_gpu()
 */
void tensor_copy_to_gpu(tensor* source, tensor* dest);

/**
 * Copies the source tensor (on the GPU) to the the destination tensor (in RAM).
 * They should have equal sizes.
 * @see tensor_copy_to_gpu(), tensor_copy_gpu_to_gpu()
 */
void tensor_copy_from_gpu(tensor* source, tensor* dest);

/**
 * Copies the source tensor to the the destination tensor (both on the GPU).
 * They should have equal sizes.
 * @see tensor_copy_to_gpu(), tensor_copy_from_gpu()
 */
void tensor_copy_gpu_to_gpu(tensor* source, tensor* dest);


#ifdef __cplusplus
}
#endif
#endif