/**
 * @file
 *
 * Central location for all copy-pad functions
 * (copying blocks of data between 3D arrays with different sizes)
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUPAD_H
#define GPUPAD_H

#include "tensor.h"
#include "gputil.h"
#include "assert.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @internal
 *
 */
void gpu_copy_pad(tensor* source, tensor* dest);

/**
 * @internal
 *
 */
void gpu_copy_unpad(tensor* source, tensor* dest); 


void gpu_copy_pad_unsafe(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2);       ///< dest size


void gpu_copy_unpad_unsafe(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2);


#ifdef __cplusplus
}
#endif
#endif