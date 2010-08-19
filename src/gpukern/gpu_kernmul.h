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
 * Kernel is in interleaved complex format (imaginary part is zero and not read, but still stored),
 * and assummed symmetric.
 * The multiplication is in-place, fftMi is overwritten by fftHi
 */
///@todo 6, 4, 3...
void gpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers);

#ifdef __cplusplus
}
#endif
#endif
