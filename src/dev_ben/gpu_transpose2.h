/**
 * @file
 * Transposition of a tensor of complex numbers
 *
 * @note Be sure not to use nvcc's -G flag, as this
 * slows down these functions by an order of magnitude
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_transpose2_h
#define gpu_transpose2_h

#ifdef __cplusplus
extern "C" {
#endif

/// 2D complex matrix transpose. Input size: N1 x N2/2 complex numbers, Output size: N2/2 x N1 complex numbers.
/// Offsets introduced to make the compatible for zero padded matrices.
void gpu_transpose_complex_offset(float *input, float *output, int N1, int N2, int offset_in, int offset_out);

/// 2D complex matrix transpose for zero padded matrix as needed in the forward FFT routine.  This routine is 'in plane'.
/// Input size: N1 x N2/2 complex numbers, Output size: N2/2 x N1 complex numbers
void gpu_transpose_complex_in_plane_fw(float *input, int N1, int N2);

/// 2D complex matrix transpose for zero padded matrix as needed in the inverse FFT routine.  This routine is 'in plane'. 
/// Input size: N1 x N2/2 complex numbers, Output size: N2/2 x N1 complex numbers
void gpu_transpose_complex_in_plane_inv(float *input, int N1, int N2);

/// Swaps the X and Z dimension of an array of complex numbers in interleaved format
// void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2);

/// Swaps the Y and Z dimension of an array of complex numbers in interleaved format
// void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
