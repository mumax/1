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

#include "gpu_safe.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Applies the analytical formulae once to obtain a new value for the magnetization based on a constant effective field.  
 * This basis function can be invoked multiple times to come to higher order convergence schemes (e.g. predictor-corrector scheme)
 */
void gpu_anal_fw_step(float dt,       ///< used time step
                      float alpha,    ///< damping constant
                      int Ntot,       ///< total number of FD cells in the simulation
                      float *m_in,    ///< array containing the input magnetization
                      float *m_out,   ///< array containing the output magnetization
                      float *h        ///< array containing the effective field (no spin torque!)
                      );


/**
 *Function to compute the mean effective field from two input arrays.  The result is stored in h1
 */
void gpu_anal_pc_mean_h(float *h1,    ///< first effective field array
                        float *h2,    ///< second effective field array
                        int Ntot      ///< total number of FD cells in the simulation
                        );


#ifdef __cplusplus
}
#endif
#endif