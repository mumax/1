#include "../libfft.h"

#include <fftw3.h>
#include <stdio.h>

#ifdef DOUBLE_PRECISSION
#define fftwf_malloc fftw_malloc
#define fftwf_free fftw_free
#define fftwf_plan_dft_r2c_3d fftw_plan_dft_r2c_3d
#define fftwf_plan_dft_c2r_3d fftw_plan_dft_c2r_3d
#define fftwf_destroy_plan fftw_destroy_plan
#define fftwf_execute fftw_execute
#define fftwf_plan fftw_plan
#endif

int _fft_initialized = 0;

void fft_init(void){
  fftwf_init_threads();
  fftwf_plan_with_nthreads(4);
  _fft_initialized = 1;
}


void fft_finalize(void){
  fftwf_cleanup_threads();
  _fft_initialized = 0;
}

void _fft_check_initialization(){
   if(!_fft_initialized){
      printf(stderr, "Bug: FFT not initialized");
      exit(2);
    }
}

real* fft_malloc(int N0, int N1, int N2){
  real* result;
  result = (real*) fftwf_malloc(N0*N1*N2 * sizeof(real));
  printf("fft_malloc_real\t(%d, %d, %d):\t%p\n", N0, N1, N2, result);
  return result;
}


void fft_free(void* data){
  printf("fft_free\t(%p)\n", data);
  fftwf_free(data);
}


void* fft_init_forward(int N0, int N1, int N2, real* source, real* dest){
    _fft_check_initialization();
    void* result = fftwf_plan_dft_r2c_3d(N0, N1, N2, source, (complex_t*)dest, FFTW_ESTIMATE); // replace by FFTW_PATIENT for super-duper performance  
    printf("fft_init_forward\t(%d, %d, %d):\t%p\n", N0, N1, N2, result);
    return result;
}


void* fft_init_backward(int N0, int N1, int N2, real* source, real* dest){
    _fft_check_initialization();
    void* result = fftwf_plan_dft_c2r_3d(N0, N1, N2, (complex_t*)source, dest, FFTW_ESTIMATE);
    printf("fft_init_backward\t(%d, %d, %d):\t%p\n", N0, N1, N2, result);
    return result;
}


void fft_destroy_plan(void* plan){
  fftwf_destroy_plan((fftwf_plan) plan);
}


void fft_execute(void* plan){
  fftwf_execute((fftwf_plan) plan);
}
