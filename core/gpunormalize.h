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
#include "param.h"
#include "gputil.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif


void gpu_normalize(param* p, tensor* m);


/// @internal
__global__ void _gpu_normalize(float* mx , float* my , float* mz);

/// @internal
__global__ void _gpu_normalize_map(float* mx , float* my , float* mz, float* normMap);

#ifdef __cplusplus
}
#endif
#endif