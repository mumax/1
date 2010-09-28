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
#include "param.h"
#include "gpukern.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @internal
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

}gpueuler;


/**
 * Makes a new euler solver.
 */
gpueuler* new_gpueuler(param* params);


/**
 * Takes one time step
 */
void gpueuler_step(gpueuler* solver,	    ///< the solver to step
                   tensor* m,		        ///< magnetization
                   tensor* h,               ///< effective field corresponding to m
                   double* totalTime        ///< pointer to the total time, is updated by the solver (deltaT is added to it)
                   );


///@internal                   
void gpu_euler_stage(float* m, float* torque, int N);

                   
#ifdef __cplusplus
}
#endif
#endif