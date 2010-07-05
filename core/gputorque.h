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

/// Overwrites h with deltaM(m, h)
void gpu_deltaM(float* m,       ///< magnetization (all 3 components, contiguously)
                float* h,       ///< effective field, to be overwritten by torque
                float alpha,    ///< damping constant
                float dt_gilb,  ///< dt * gilbert factor
                int N           ///< length of each of the components of m, h (1/3 of their total length)
                );


/// @internal
__global__ void _gpu_deltaM(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float alpha, float dt_gilb);


#ifdef __cplusplus
}
#endif
#endif