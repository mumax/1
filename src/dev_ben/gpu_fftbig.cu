#include <cufft.h>
#include "gputil.h"
#include <cufft.h>
#include "gpu_safe.h"
#include "gpu_fft5.h"
#include "gpu_fftbig.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif

int MAXSIZE = 16*1024;

void init_bigfft(bigfft* plan, int size_fft_in, int size_fft_out, cufftType type, int Nffts){

/*  plan = (bigfft *) malloc(sizeof(bigfft));*/
  init_batch_bigfft (plan, size_fft_in, size_fft_out, Nffts);
  
  gpu_safefft( cufftPlan1d( &plan->Plan_1, size_fft_in, type, plan->batch[0]) );
  if ( plan->batch[plan->Nbatch-1] != plan->batch[0] )
    gpu_safefft( cufftPlan1d( &plan->Plan_2, size_fft_in, type, plan->batch[plan->Nbatch-1]) );
  
  plan->batch_Plans  = (cufftHandle *) calloc(plan->Nbatch, sizeof(cufftHandle));

  for (int i=0; i<plan->Nbatch; i++)
    if ( plan->batch[i] == plan->batch[0] )
      plan->batch_Plans[i] = plan->Plan_1;
    else
      plan->batch_Plans[i] = plan->Plan_2;

  return;
}


void init_batch_bigfft(bigfft *plan, int size_fft_in, int size_fft_out, int Nffts){

  int K = get_factor_to_stride(size_fft_in);
  int max = MAXSIZE/size_fft_in/K;

  if (MAXSIZE < size_fft_in*K){
    printf("size %d does not fit the fft batch!", size_fft_in);
    printf("K*FFT_array should be smaller than %d AND K*FFT_array modulo %d should be zero\n", MAXSIZE, gpu_stride_float());
    printf("choose different dimensions (i.e. size of FFT_array) to meet this criterium.");
    abort();
  }

  int Nbatch = Nffts/K/max + (Nffts%(K*max) + K*max - 1)/(K*max);
  plan->Nbatch = Nbatch;
  plan->batch  = (int*) calloc(Nbatch, sizeof(int));
  plan->batch_index_in  = (int*) calloc(Nbatch, sizeof(int));
  plan->batch_index_out = (int*) calloc(Nbatch, sizeof(int));

  int left = Nffts;
  int cnt=0;
  while (left >= K*max){
    plan->batch[cnt] = K*max;
    left -=K*max;
    cnt++;
  }
  if (left!=0)
    plan->batch[cnt] = left;
    
  plan->batch_index_in [0] = 0;
  plan->batch_index_out[0] = 0;
  for (int i=1; i< Nbatch; i++){
    plan->batch_index_in [i] = plan->batch_index_in [i-1] + plan->batch[i-1]*size_fft_in;
    plan->batch_index_out[i] = plan->batch_index_out[i-1] + plan->batch[i-1]*size_fft_out;
  }
  
  return;
}


int get_factor_to_stride(int size_fft){

  int K = 1;
  int stride = gpu_stride_float();
  while (size_fft % (stride/K) !=0)
    K*=2;

  return (K);
}


void bigfft_execR2C(bigfft* plan, cufftReal* input, cufftComplex* output){

  for (int i=0; i<plan->Nbatch; i++){
    int index_in  = plan->batch_index_in [i];
    int index_out = plan->batch_index_out[i]/2;
//     printf("R2C: %d, %d\n", index_in, index_out);
    gpu_safefft( cufftExecR2C(plan->batch_Plans[i], input + index_in,  output + index_out) );
  }

  return;
}


void bigfft_execC2R(bigfft* plan, cufftComplex* input, cufftReal* output){

  for (int i=0; i<plan->Nbatch; i++){
    int index_in  = plan->batch_index_in [i]/2;
    int index_out = plan->batch_index_out[i];
//     printf("C2R: %d, %d\n", index_in, index_out);
    gpu_safefft( cufftExecC2R(plan->batch_Plans[i], input + index_in,  output + index_out) );
  }

  return;
}

void bigfft_execC2C(bigfft* plan, cufftComplex* input, cufftComplex* output, int direction){
  for (int i=0; i<plan->Nbatch; i++){
    int index_in  = plan->batch_index_in [i];
    int index_out = plan->batch_index_out[i];
//     printf("C2C: %d, %d\n", index_in, index_out);
    gpu_safefft( cufftExecC2C(plan->batch_Plans[i], input + index_in,  output + index_out, direction) );
  }
  return;
}


#ifdef __cplusplus
}
#endif