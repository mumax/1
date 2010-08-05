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
