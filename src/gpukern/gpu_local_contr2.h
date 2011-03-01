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
 * @author Arne Vansteenkiste, Ben Van de Wiele
 *
 */
#ifndef gpu_local_contr2_h
#define gpu_local_contr2_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Adds the local contributions to the effective field.
 */
void gpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes);


/**
 * Adds the local contributions to the effective field and to the energy density.
 */
void gpu_add_local_fields_H_and_phi (float* m, float* h, float *phi, int N, float* Hext, int anisType, float* anisK, float* anisAxes);

#ifdef __cplusplus
}
#endif
#endif