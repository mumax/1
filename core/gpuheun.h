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
 * @see new_gpuheun2
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
  
}gpuheun2;


// gpuheun2* new_gpuheun2(int* size,           ///< 3D size of magnetization
//                       tensor* kernel,       ///< convolution kernel describing the effective field. size: 2*N0 x 2*N1 x 2*N2
//                       float* hExt           ///< external field
//                       );

/**
 * Makes a new heun solver.
 */                      
gpuheun2* new_gpuheun2(param* p);

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
void gpuheun2_step(gpuheun2* solver, tensor* m, tensor* h, double* totalTime);

      

#ifdef __cplusplus
}
#endif
#endif