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
 * @note All functions are "safe", i.e. they will never fail silently but print
 * an error report and abort() when an error occurs.
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

/// Maximum size of a CUFFT FFT
#define MAX_FFTSIZE (8*1024*1024)

/// @internal
/// The batch is split into plan1 (with the largest possible size),
/// which is executed many times, and plan2 (with the remainder of the elements),
/// which is executed once.
/// plan1->size = MAX_FFTSIZE, executed N / MAX_FFTSIZE times
/// plan2->size = N % MAX_FFTSIZE, executed once.
typedef struct{
  int nPlan1;         ///< For an FFT with N elements, N = nPlan1 x plan1->size + plan2->size
  cufftHandle plan1;
  int nPlan2;         ///< Execute plan2 nPlan2 times (O or 1). If N is divisible by MAX_FFTSIZE, plan2 does not need to be executed.
  cufftHandle plan2;
  int size;           ///< Logical size of the individual transforms (in complex numbers)
  int maxBatch;       ///< The maximum number of batches in a plan, plan1 has this number of batches.
}bigfft;

void init_bigfft(bigfft* target, int size, cufftType type, int batch);

void bigfft_execR2C(bigfft* plan, cufftReal* input, cufftComplex* output);

void bigfft_execC2R(bigfft* plan, cufftComplex* input, cufftReal* output);

void bigfft_execC2C(bigfft* plan, cufftComplex* input, cufftComplex* output, int direction);


#ifdef __cplusplus
}
#endif
#endif
