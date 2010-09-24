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

#ifdef __cplusplus
}
#endif
#endif
