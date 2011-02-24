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
 * Transposition of a tensor of complex numbers
 *
 * @todo This implementation is way too slow,
 * see the transpose example in the nvidia SDK for a better way.
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_transpose_h
#define cpu_transpose_h

#ifdef __cplusplus
extern "C" {
#endif


/// Swaps the X and Z dimension of an array of complex numbers in interleaved format
void cpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2);

/// Swaps the Y and Z dimension of an array of complex numbers in interleaved format
void cpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
