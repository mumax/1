/**
 * @file
 *
 * This file implements the classical computation of the exchange field on cpu.
 * The procedure checks for every component of the field if the exchange contribution is allready added in the convolution.  
 * If not, the contribution is added here.  A distinction is made between 2D and 3D geometries.
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef CPU_EXCH_H
#define CPU_EXCH_H

#include "param.h"
#include "gpukern.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_addExch (tensor *m,    ///> input magnetization tensor
                  tensor *h,    ///> output field tensor
                  param *p      ///> simulation parameters
                 );

void cpu_addExch_2D_geometry (float *m,     ///> input magnetization tensor
                              float *h,     ///> output field tensor
                              param *p     ///> simulation parameters
                              );

void cpu_addExch_3D_geometry (float *m,     ///> input magnetization tensor
                              float *h,     ///> output field tensor
                              param *p      ///> simulation parameters
                              );
             
#ifdef __cplusplus
}
#endif
#endif