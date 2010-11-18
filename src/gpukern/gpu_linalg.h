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

#ifdef __cplusplus
}
#endif
#endif
