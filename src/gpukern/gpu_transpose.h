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
#ifndef gpu_transpose_h
#define gpu_transpose_h

#ifdef __cplusplus
extern "C" {
#endif

/// 2D transpose
void gpu_transpose(float *odata, float *idata, int size_x, int size_y);

/// Swaps the X and Z dimension of an array of complex numbers in interleaved format
void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2);

/// Swaps the Y and Z dimension of an array of complex numbers in interleaved format
void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
