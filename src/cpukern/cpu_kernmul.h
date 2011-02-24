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
#ifndef cpu_kernmul_h
#define cpu_kernmul_h

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @internal
 * Extract only the real parts from an interleaved complex array.
 */
//void cpu_extract_real(float* complex, float* real, int NReal);


/**
 * @internal 
 * Kernel is symmetric.
 * The multiplication is in-place, fftMi is overwritten by fftHi
 */
void cpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers);
/**
 * @internal 
 * Kernel is symmetric and Kxy = Kxz = 0.
 * The multiplication is in-place, fftMi is overwritten by fftHi
 */
void cpu_kernelmul4(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz,
                    int nRealNumbers);

#ifdef __cplusplus
}
#endif
#endif
