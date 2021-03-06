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
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_anis_h
#define gpu_anis_h

#ifdef __cplusplus
extern "C" {
#endif

/// Adds a linear anisotropy contribution to h:
/// h_i += Sum_i k_ij * m_j
/// Used for edge corrections.
void gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                            float* mx, float* my, float* mz,
                            float* kxx, float* kyy, float* kzz,
                            float* kyz, float* kxz, float* kxy,
                            int N);

#ifdef __cplusplus
}
#endif
#endif
