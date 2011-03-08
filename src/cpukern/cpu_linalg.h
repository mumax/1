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
 * General linear algebra functions
 *
 * @author Arne Vansteenkiste
 */
#ifndef cpu_linalg_h
#define cpu_linalg_h

#ifdef __cplusplus
extern "C" {
#endif


/// Adds array b to a
void cpu_add(float* a, float* b, int N);

/// a[i] += cnst * b[i]
void cpu_madd(float* a, float cnst, float* b, int N);

/// Adds a constant to array a
void cpu_add_constant(float* a, float cnst, int N);

/// Linear combination: a = a*weightA + b*weightB
void cpu_linear_combination(float* a, float* b, float weightA, float weightB, int N);

/// result = a*(vector1 'dot' vector2), vector1 and vector2 have 3 components
void cpu_scale_dot_product(float* result, float *vector1, float *vector2, float a, int N);

#ifdef __cplusplus
}
#endif
#endif
