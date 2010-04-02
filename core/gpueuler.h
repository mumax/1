/**
 * @file
 *
 * This file implements a trivial Euler algorithm for solving the LL equation.
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUEULER_H
#define GPUEULER_H

#include "tensor.h"
#include "gpuconv1.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpueuler
 */
typedef struct{
  
  int* size;
  int N;
  
  float* m;
  int len_m;
  
  float* h;
  int len_h;
  
  gpuconv1* convplan;
  
}gpueuler;

/**
 * Makes a new euler solver.
 */
gpueuler* new_gpueuler(int N0,		///< X-size of magnetization
		       int N1,		///< Y-size of magnetization
		       int N2, 		///< Z-size of magnetization
		       tensor* kernel	///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
		       );

/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpueuler_storem
 */
void gpueuler_loadm(gpueuler* euler, tensor* m);

/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpueuler_loadm
 */
void gpueuler_storem(gpueuler* euler, tensor* m);

/**
 * Takes one time step
 */
void gpueuler_step(gpueuler* solver,	///< the solver to step
		   float dt		///< time step (internal units).
		   );

#ifdef __cplusplus
}
#endif
#endif