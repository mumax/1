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
#ifndef gpu_zeropad_h
#define gpu_zeropad_h

#ifdef __cplusplus
extern "C" {
#endif


void gpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2         ///< dest size
                         );      


void gpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2         ///< dest size
                         );
                         
void gpu_copy_pad2D(float* source, float* dest,
                         int S1, int S2,
                         int D1, int D2);

#ifdef __cplusplus
}
#endif
#endif
