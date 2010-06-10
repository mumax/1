/**
 * @file
 *
 * This file implements a Heun algorithm for solving the LL equation.
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUHEUN2_H
#define GPUHEUN2_H

#include "tensor.h"
#include "gpuconv2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpuheun2
 */
typedef struct{
  
  tensor* m;
  tensor* m0;
  
  tensor* h;
  tensor* torque0;
  
  tensor* m_comp[3];
  tensor* m0_comp[3];

  tensor* h_comp[3];
  tensor* torque0_comp[3];
  
  gpuconv2* convplan;
  
  float* hExt;
  
}gpuheun2;

/**
 * Makes a new heun solver.
 */
gpuheun2* new_gpuheun2(int* size,		///< 3D size of magnetization
                      tensor* kernel,	///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
                      float* hExt	///< external field
                      );

/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpuheun_storem
 */
void gpuheun2_loadm(gpuheun2* heun, tensor* m);

/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpuheun_loadm, gpuheun_storeh
 */
void gpuheun2_storem(gpuheun2* heun, tensor* m);

/**
 * Copies the last magnetic field from the solver to the RAM.
 * @see: gpuheun_storem
 */
void gpuheun2_storeh(gpuheun2* heun, tensor* h);


/**
 * Takes one time step
 */
void gpuheun2_step(gpuheun2* solver,        ///< the solver to step
                   float dt,                ///< time step (internal units)
                   float alpha              ///< damping
                   );

#ifdef __cplusplus
}
#endif
#endif