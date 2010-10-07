/**
 * @file
 * Accesses the GPU's hardware properties
 *
 * @author Arne Vansteenkiste
 */
#ifndef gpu_properties_h
#define gpu_properties_h

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


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
 * code (Go, in my case).
 */
void* gpu_getproperties(void);


/// Prints the properties of the used GPU
void gpu_print_properties(FILE* out  ///< stream to print to
);


/// Prints to stdout
/// @see print_device_properties()
void gpu_print_properties_stdout(void);


#ifdef __cplusplus
}
#endif
#endif
