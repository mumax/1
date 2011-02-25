/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 *
 * This file implements the addition of the local contributions to the effective field.  These are
 *    - Applied field
 *    - anisotropy field
 *
 *    @todo thermal field, magnetoelastic field
 *
 * @author Ben Van de Wiele
 *
 */
#ifndef GPU_LOCAL_CONTR_H
#define GPU_LOCAL_CONTR_H

#ifdef __cplusplus
extern "C" {
#endif
/**
 * list of parameter data arrays required for computations on gpu
 */
typedef struct{

  float *anisK;     ///> anisotropy constants
  float *anisAxes;  ///> float array defining the 
  
}dev_par;


/**
 * Adds the local contributions to the effective field.
 */
void gpu_add_local_contr (float *m,         ///> magnetization
                          float *h,         ///> effective field
                          int Ntot,         ///> total number of FD cells
                          float *Hext,      ///> uniform external field
                          int anisType,     ///> anisotropy type
                          dev_par *p_dev    ///> list of parameter data required for computations on gpu
                          );


/**
 * Initializes the parameters on gpu required for local field computations
 */
dev_par* init_par_on_dev(int anisType,      ///> anisotropy type
                         float *anisK,      ///> array containing the anisotropy constant(s): uniaxial K_u; cubic K_1 K_2.
                         float *defAxes     ///> 3 float array: anisotropy axis (uniaxial anisotropy) or euler angles -in radiants- defining the cubic uniaxial axes (cubic anisotropy)
                         );


/**
 * Frees the parameters on gpu required for local field computations
 */
void destroy_par_on_dev(dev_par *p_dev,     ///> list of parameter data required for computations on gpu
                        int anisType        ///> anisotropy type
                        );




#ifdef __cplusplus
}
#endif
#endif