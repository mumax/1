/**
 * @file
 * The classical 4th order Runge-Kutta algorithm for solving the LL equation.
 *
 * @note Does not work yet
 *
 * @note When writing out of allocated memory bounds, 
 * the GPU of course does not throw a segmentation fault. 
 * However, I had one case where a copy operation AFTER faulty
 * memory acces failed with "the launch timed out and was terminated",
 * and also my screen went black for a split-second. Writing
 * out of bounds can thus actually affect your graphics!
 *
 * @todo we do not need 4 k arrays: just one where we accumulate the total k
 * first calc k_i in shared memory, use it to set the next m_i+1
 * then add it to the global k array with the correct weigth
 * the last k_i is not even added to that array but immediately to m
 *
 * @author Arne Vansteenkiste
 */
#ifndef GPURK4_H
#define GPURK4_H

#include "tensor.h"
#include "gpuconv1.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpurk4
 */
typedef struct{
  
  int* size;
  int N;
  
  float* m;
  float* m0;
  float** k;
  int len_m;
  int len_m_comp;
  
  float* h;
  int len_h;
  
  gpuconv1* convplan;
  
}gpurk4;

/**
 * Makes a new euler solver.
 */
gpurk4* new_gpurk4(int N0,		///< X-size of magnetization
		       int N1,		///< Y-size of magnetization
		       int N2, 		///< Z-size of magnetization
		       tensor* kernel	///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
		       );

/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpurk4_storem
 */
void gpurk4_loadm(gpurk4* rk, tensor* m);

/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpurk4_loadm
 */
void gpurk4_storem(gpurk4* rk, tensor* m);

/**
 * Takes one time step
 */
void gpurk4_step(gpurk4* rk,		///< the solver to step
		   float dt		///< time step (internal units).
		   );

#ifdef __cplusplus
}
#endif
#endif