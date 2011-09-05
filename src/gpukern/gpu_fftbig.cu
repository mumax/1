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
 * @author Ben Van de Wiele
 */
#include <cufft.h>
#include <cufft.h>
#include "gpu_safe.h"
#include "gpu_fftbig.h"
#include "gpu_conf.h"
#include "gpu_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

int MAXSIZE = 8*1024*1024;

void init_bigfft(bigfft* plan, int size_fft, int stride_in, int stride_out, cufftType type, int Nffts){

  init_batch_bigfft (plan, stride_in, stride_out, Nffts);
  
  gpu_safefft( cufftPlan1d( &plan->Plan_1, size_fft, type, plan->batch[0]) );
//   gpu_safefft( cufftSetCompatibilityMode(plan->Plan_1, CUFFT_COMPATIBILITY_NATIVE) );
  if ( plan->batch[plan->Nbatch-1] != plan->batch[0] ){
    gpu_safefft( cufftPlan1d( &plan->Plan_2, size_fft, type, plan->batch[plan->Nbatch-1]) );
//     gpu_safefft( cufftSetCompatibilityMode(plan->Plan_2, CUFFT_COMPATIBILITY_NATIVE) );
  }
  else 
    plan->Plan_2 = plan->Plan_1;

  plan->batch_Plans  = (cufftHandle *) calloc(plan->Nbatch, sizeof(cufftHandle));
  for (int i=0; i<plan->Nbatch-1; i++)
    plan->batch_Plans[i] = plan->Plan_1;
  plan->batch_Plans[plan->Nbatch-1] = plan->Plan_2;

  return;
}

void init_batch_bigfft(bigfft *plan, int stride_in, int stride_out, int Nffts){

  int K = get_factor_to_stride(stride_in);
  int max = MAXSIZE/stride_in/K;

  if (MAXSIZE < stride_in*K){
    printf("size %d does not fit the fft batch!\n", stride_in);
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
    plan->batch_index_in [i] = plan->batch_index_in [i-1] + plan->batch[i-1]*stride_in;
    plan->batch_index_out[i] = plan->batch_index_out[i-1] + plan->batch[i-1]*stride_out;
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
/*    int index_in  = plan->batch_index_in [i];
    int index_out = plan->batch_index_out[i]/2;*/
    int index_in  = plan->batch_index_in [i];
    int index_out = plan->batch_index_out[i]/2;
//     printf("R2C: %d, %d, %d\n", i, index_in, index_out);
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

void delete_bigfft(bigfft *plan){

  if (plan->Plan_2 != plan->Plan_1)
    cufftDestroy(plan->Plan_2);
  cufftDestroy(plan->Plan_1);
  
  free (plan->batch_Plans);
  free (plan->batch_index_in);
  free (plan->batch_index_out);
  free (plan->batch);
  
  free (plan);
  
  return;
}

#ifdef __cplusplus
}
#endif