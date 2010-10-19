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


#ifdef __cplusplus
}
#endif
#endif
