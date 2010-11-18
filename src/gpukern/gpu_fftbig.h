/**
 * @file
 *
 * @author Ben Van de Wiele
 */
#ifndef GPU_FFTBIG_H
#define GPU_FFTBIG_H


#ifdef __cplusplus
extern "C" {
#endif


#include <cufft.h>

typedef struct{

  int Nbatch;
  int *batch;
  int *batch_index_in;
  int *batch_index_out;
  
  cufftHandle Plan_1;
  cufftHandle Plan_2;

  cufftHandle *batch_Plans;     ///< 1D real-to-complex plan for Z-direction

} bigfft;


void init_bigfft(bigfft* plan,
                 int size_fft,
                 int stride_in,
                 int stride_out,
                 cufftType type,
                 int Nffts
                 );
                 
void init_batch_bigfft(bigfft *plan, 
                       int stride_in, 
                       int stride_out, 
                       int Nffts
                       );

int get_factor_to_stride(int size_fft);

void bigfft_execR2C(bigfft* plan,
                    cufftReal* input,
                    cufftComplex* output
                    );
                  
void bigfft_execC2R(bigfft* plan,
                    cufftComplex* input,
                    cufftReal* output
                    );

void bigfft_execC2C(bigfft* plan,
                    cufftComplex* input,
                    cufftComplex* output,
                    int direction);

void delete_bigfft(bigfft *plan);

#ifdef __cplusplus
}
#endif
#endif