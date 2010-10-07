/**
 * @file
 *
 * This file implements the forward and predictor/corrector semi-analytical time stepping scheme for solving the LL equation.
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef GPUANAL_H
#define GPUANAL_H

#include "param.h"
#include "gputil.h"
#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The internal data of the solver.
 * @see new_gpuanal1
 */
typedef struct{

  param *params;

}gpuanalfw;

typedef struct{

  param *params;
  tensor *m2;
  tensor *h2;

}gpuanalpc;



void gpu_anal_fw_step(param *p, tensor *m_in, tensor *m_out, tensor *h);

__global__ void _gpu_anal_fw_step (float *minx, float *miny, float *minz, float *moutx, float *mouty, float *moutz, float *hx, float *hy, float *hz, float dt, float alpha, int N);


void gpu_anal_pc_mean_h(tensor *h1, tensor *h2);

__global__ void _gpu_anal_pc_meah_h (float *h1, float *h2, int N);

gpuanalfw* new_gpuanalfw(param* p);

gpuanalpc* new_gpuanalpc(param* p);


#ifdef __cplusplus
}
#endif
#endif