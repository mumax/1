/**
 * @file
 *
 *
 *
 * @author Arne Vansteenkiste, Ben Van de Wiele
 *
 */
#ifndef KERNEL_H
#define KERNEL_H


#ifdef __cplusplus
extern "C" {
#endif

#include "gpu_fft5.h"
#include "tensor.h"
#include "param.h"
#include "gpu_micromag3d_kernel.h"
#include "gpu_micromag2d_kernel.h"

tensor* new_kernel(param*);


#ifdef __cplusplus
}
#endif
#endif