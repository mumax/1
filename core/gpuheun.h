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
#include "param.h"
#include "gpuconv2.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpuheun
 */
typedef struct{

  param* params;
  
  tensor* m;
  tensor* m0;
  tensor* h;
  tensor* torque0;
  
  tensor* mComp[3];
  tensor* m0Comp[3];
  tensor* hComp[3];
  tensor* torque0Comp[3];

  int stage;
  
}gpuheun;


/**
 * Makes a new heun solver.
 */                      
gpuheun* new_gpuheun(param* p);


/**
 * Copies a magnetization configuration in the solver, e.g. the initial magnetization.
 * @see: gpuheun_storem
 */
//void gpuheun_loadm(gpuheun* heun, tensor* m);


/**
 * Copies a magnetization configuration from the solver to the RAM, e.g. the magnetization after a number of time steps.
 * @see: gpuheun_loadm, gpuheun_storeh
 */
//void gpuheun_storem(gpuheun* heun, tensor* m);

/**
 * Copies the last magnetic field from the solver to the RAM.
 * @see: gpuheun_storem
 */
//void gpuheun_storeh(gpuheun* heun, tensor* h);


/**
 * Takes one time step
 */
void gpuheun_step(gpuheun* solver, tensor* m, tensor* h, double* totalTime);

      

#ifdef __cplusplus
}
#endif
#endif