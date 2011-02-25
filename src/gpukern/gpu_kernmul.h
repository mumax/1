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
 * @author Ben Van de Wiele
 * @author Arne Vansteenkiste
 */
#ifndef gpu_kernmul_h
#define gpu_kernmul_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @internal
 * Extract only the real parts from an interleaved complex array.
 */
void gpu_extract_real(float* complex, float* real, int NReal); 


/**
 * @internal 
 * FFT'ed Kernel is purely real and assummed symmetric in Kij.
 * The multiplication is in-place, fftMi is overwritten by fftHi
 */
///@todo 6, 4, 3...
void gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers);

void gpu_kernelmul4(float *fftMx,  float *fftMy, float *fftMz, 
                    float *fftKxx, float *fftKyy, float *fftKzz, 
                    float *fftKyz, 
                    int nRealNumbers
                    );

void gpu_kernelmul3(float *fftMy, float *fftMz, 
                    float *fftKyy, float *fftKzz, 
                    float *fftKyz, 
                    int nRealNumbers
                    );

void gpu_kernelmul_biot_savart3D(float* fftJx, float* fftJy, float* fftJz,
                                 float* fftKx, float* fftKy, float* fftKz,
                                 int nRealNumbers);
                                 
void gpu_kernelmul_biot_savart3DNx1(float* fftJx, float* fftJy, float* fftJz,
                                    float* fftKy, float* fftKz,
                                    int nRealNumbers);
                                    
void gpu_kernelmul_biot_savart2D(float* fftJx,  float* fftJy,  float* fftJz,
                                 float* fftKy, float* fftKz,
                                 int nRealNumbers);
#ifdef __cplusplus
}
#endif
#endif
