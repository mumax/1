/**
 * @file
 *
 * The CUDA "CUFFT" FFT library allows only transforms up to 8 million elements.
 * When using 1D batched transforms, this is however insufficient for relatively
 * large simulations (1024 x 1024 is about the maximum). The solution is to run
 * several smaller batches sequentially, but there is the question about how to
 * split up the batch without sacrificing performance nor generality.
 * 
 * Here we implement "big" 1D FFT batches in a transparent way. We create a big 
 * fft plan and just execute it. Efficiently plitting into smaller batches is done
 * internally so we don't need to worry about it outside of this file.
 *
 * @note The FFT's are initiated with CUFFT "native" compatibility, which optimizes
 * performance.
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_bigfft_h
#define gpu_bigfft_h

#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct{
  cufftHandle plan;
}bigfft;

void init_bigfftR2C(bigfft* target, int size, int batch);

void init_bigfftC2R(bigfft* target, int size, int batch);

void init_bigfftC2C(bigfft* target, int size, int batch);

void bigfft_execR2C(bigfft* plan, float* input, float* output);

void bigfft_execC2R(bigfft* plan, float* input, float* output);

void bigfft_execC2C(bigfft* plan, float* input, float* output, int direction);


#ifdef __cplusplus
}
#endif
#endif
