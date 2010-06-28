/**
 * @file
 *
 * Torque functions
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUTORQUE_H
#define GPUTORQUE_H

#include "gputil.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Overwrites h with torque(m, h)
void gpu_torque(float* m,       ///< magnetization (all 3 components, contiguously)
                float* h,       ///< effective field, to be overwritten by torque
                float alpha,    ///< damping constant
                int N           ///< length of each of the components of m, h (1/3 of their total length)
                );


/// @internal
__global__ void _gpu_torque(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float alpha);


#ifdef __cplusplus
}
#endif
#endif