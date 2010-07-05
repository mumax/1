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

#include "debug.h"
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
  
  tensor* m; // todo: get rid of
  tensor* m0;
  tensor* h; // todo: get rid of
  tensor* torque0;
  
  tensor* mComp[3]; // todo: get rid of
  tensor* m0Comp[3];
  tensor* hComp[3]; // todo: get rid of
  tensor* torque0Comp[3];

  int stage;
  
}gpuheun;


/**
 * Makes a new heun solver.
 */                      
gpuheun* new_gpuheun(param* p);


/**
 * Takes one time step
 */
void gpu_heun_step(gpuheun* solver, tensor* m, tensor* h, double* totalTime);

///@internal
void gpuheun_stage0_(float* m, float* torque, int N);

///@internal
void gpuheun_stage1_(float* m, float* torque, int N);

///@internal
void gpu_linear_combination(float* a, float* b, float weightA, float weightB, int N);

#ifdef __cplusplus
}
#endif
#endif