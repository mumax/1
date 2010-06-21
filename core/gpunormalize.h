/**
 * @file
 *
 * Normalization of m, possible space-dependent
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUNORMALIZE_H
#define GPUNORMALIZE_H

#include "tensor.h"
#include "gputil.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


void gpu_normalize(tensor* m);


/// @internal
__global__ void _gpu_normalize(float* mx , float* my , float* mz);


#ifdef __cplusplus
}
#endif
#endif