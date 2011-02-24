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
#ifndef gpu_linalg_h
#define gpu_linalg_h

#ifdef __cplusplus
extern "C" {
#endif


/// Adds array b to a
void gpu_add(float* a, float* b, int N);

/// a[i] += cnst * b[i]
void gpu_madd(float* a, float cnst, float* b, int N);

/// a[i] += b[i] * c[i]
void gpu_madd2(float* a, float* b, float* c, int N);

/// Adds a constant to array a
void gpu_add_constant(float* a, float cnst, int N);

/// Linear combination: a = a*weightA + b*weightB
void gpu_linear_combination(float* a, float* b, float weightA, float weightB, int N);

/// @todo rename: ADD
/// Linear combination: result[i] += sum_j weights[j] * vectors[j][i]
/// i = 0..NElem
/// j = 0..NVectors
void gpu_linear_combination_many(float* result, float** vectors, float* weights, int NVectors, int NElem);

#ifdef __cplusplus
}
#endif
#endif
