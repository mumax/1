/**
 * @file
 *
 * This file implements the forward semi-analytical time stepping scheme for solving the LL equation.
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef GPUANAL1_H
#define GPUANAL1_H

#include "tensor.h"
#include "gpuconv1.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpuanal1
 */
typedef struct{
  
  int* size;
  int N;
  
  float* m;
  int len_m;
  
  float* h;
  int len_h;
  
  gpuconv1* convplan;
  
}gpuanal1;

/**
 * Makes a new anal1 solver.
 */
gpuanal1* new_gpuanal1(int N0,		///< X-size of magnetization
		       int N1,		///< Y-size of magnetization
		       int N2, 		///< Z-size of magnetization
		       tensor* kernel	///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
		       );

/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpuanal1_storem
 */
void gpuanal1_loadm(gpuanal1* anal1, tensor* m);

/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpuanal1_loadm
 */
void gpuanal1_storem(gpuanal1* anal1, tensor* m);

/**
 * Takes one time step
 */
void gpuanal1_step(gpuanal1* solver,	///< the solver to step
		   float dt		///< time step (internal units).
		   );

#ifdef __cplusplus
}
#endif
#endif