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
void gpu_transpose_complex_offset2(float *input, float *output, int N1, int N2, int offset_in, int offset_out, int N, int stride1, int stride2);

/// 2D complex matrix transpose for zero padded matrix as needed in the forward FFT routine.  This routine is 'in plane'.
/// Input size: N1 x N2/2 complex numbers, Output size: N2/2 x N1 complex numbers
void gpu_transpose_complex_in_plane_fw(float *input, int N1, int N2);

/// 2D complex matrix transpose for zero padded matrix as needed in the inverse FFT routine.  This routine is 'in plane'. 
/// Input size: N1 x N2/2 complex numbers, Output size: N2/2 x N1 complex numbers
void gpu_transpose_complex_in_plane_inv(float *input, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
