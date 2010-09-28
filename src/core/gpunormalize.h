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
#include "gpukern.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Normalizes the magnetization (or any other tensor).
 */
void gpu_normalize(param* p,    ///< parameters indicate whether normalization is space-dependent and store msatMap if applicable.
                   tensor* m    ///< 3 x N0 x N1 x N2 tensor to normalize
                   );

///@internal
void gpu_normalize_uniform(float* m, int N);

///@internal
void gpu_normalize_map(float* m, float* map, int N);


/// @internal uniform normalization
__global__ void _gpu_normalize(float* mx , float* my , float* mz);

/// @internal space-dependent normalization
__global__ void _gpu_normalize_map(float* mx , float* my , float* mz, float* normMap);

#ifdef __cplusplus
}
#endif
#endif