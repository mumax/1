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
#ifndef gpu_torque_h
#define gpu_torque_h

#ifdef __cplusplus
extern "C" {
#endif



/// Overwrites h with deltaM(m, h)
void gpu_deltaM(float* m,       ///< magnetization (all 3 components, contiguously)
                float* h,       ///< effective field, to be overwritten by torque
                float alpha,    ///< damping constant
                float dt_gilb,  ///< dt * gilbert factor
                int N           ///< length of each of the components of m, h (1/3 of m's total length)
                );

#ifdef __cplusplus
}
#endif
#endif
