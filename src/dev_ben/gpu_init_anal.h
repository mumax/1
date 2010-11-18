/**
 * @file
 *
 * This file implements the forward and predictor/corrector semi-analytical time stepping scheme for solving the LL equation.
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef GPUINIT_ANAL_H
#define GPUINIT_ANAL_H

#include "param.h"
// #include "gputil.h"
#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 */
typedef struct{

  param *params;

}gpuanalfw;

typedef struct{

  param *params;
  tensor *m2;
  tensor *h2;

}gpuanalpc;


gpuanalfw* new_gpuanalfw(param* p);

gpuanalpc* new_gpuanalpc(param* p);


#ifdef __cplusplus
}
#endif
#endif