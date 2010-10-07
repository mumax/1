/**
 * @file
 * This file provides some common functions for the GPU, like allocating arrays on it...
 *
 * @author Arne Vansteenkiste
 */
#ifndef gpu_mem_h
#define gpu_mem_h

#ifdef __cplusplus
extern "C" {
#endif



/**
 * Allocates an array of floats on the GPU
 * @see new_ram_array()
 */
float* new_gpu_array(int size	///< number of floats
                    );

/// Frees an array allocated by new_gpu_array                    
void free_gpu_array(float* ptr);

/// Returns how much bytes are allocated on the GPU                    
unsigned long long int gpu_usedmem();
                    
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
 */
int gpu_stride_float();


/**
 * @internal For debugging, it is handy to use a smaller-than-optimal stride;
 * this prevents small test data to be padded to huge proportions. To reset
 * to the intrinsic machine stride, set the value to -1.
 */
void gpu_override_stride(int nFloats    ///< The stride (in number of floats) to use instead of the real, GPU-dependent stride.
                        );
                        
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
// int gpu_pad_to_stride(int nFloats);


#define DIR_TO 1
#define DIR_ON 2
#define DIR_FROM 3

/**
 * Copies floats to, on or from the GPU, depending on the given direction
 * @see DIR_TO DIR_ON DIR_FROM memcpy_to_gpu() memcpy_from_gpu(), memcpy_on_gpu()
 */
void memcpy_gpu_dir(float* source, float* dest, int nElements, int direction);

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
 * Set a range of floats on the GPU to zero.
 */
void gpu_zero(float* data,	///< data pointer on the GPU
              int nElements	///< number of floats (not bytes) to be zeroed
              );


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


#ifdef __cplusplus
}
#endif
#endif