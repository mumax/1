/**
 * @file
 *
 * This file implements a Heun algorithm for solving the LL equation.
 *
 * @author Arne Vansteenkiste
 *
 */
#ifndef GPUHEUN_H
#define GPUHEUN_H

#include "tensor.h"
#include "gpuconv1.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpuheun
 */
typedef struct{
  
  int* size;
  int N;
  
  float* m;
  float* m0;
  int len_m;
  
  float* h;
  float* torque0;
  
  float** m_comp;
  float** m0_comp;
  int len_m_comp;
    
  float** h_comp;
  float** torque0_comp;
  
  gpuconv1* convplan;
  
}gpuheun;

/**
 * Makes a new heun solver.
 */
gpuheun* new_gpuheun(int N0,		///< X-size of magnetization
		       int N1,		///< Y-size of magnetization
		       int N2, 		///< Z-size of magnetization
		       tensor* kernel	///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
		       );

/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpuheun_storem
 */
void gpuheun_loadm(gpuheun* heun, tensor* m);

/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpuheun_loadm
 */
void gpuheun_storem(gpuheun* heun, tensor* m);

/**
 * Takes one time step
 */
void gpuheun_step(gpuheun* solver,	///< the solver to step
		   float dt		///< time step (internal units).
		   );

#ifdef __cplusplus
}
#endif
#endif