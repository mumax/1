#include "../libfft.h"
#include <cufft.h>

#include <stdio.h>

/** An FFTW plan "knows" everything it needs to know, including its source and destination arrays and direction (forward/backward). We can thus just call execute(fftw_plan). A CUDA plan, on the other hand, does not store this information. Therefore, we wrap this information in a cudaPlan, which is similar to an FFTW_plan. This allows us to call execute(plan), regardless of the plan being an FFTW or CUDA plan.
*/
typedef struct{
  cufftHandle handle;
  
  real* source;
  real* dest;
  
  real* device;
  
  int N0, N1, N2;
  int direction;
} cudaPlan;


void fft_init(void){
}


void fft_finalize(void){
}


real* fft_malloc(int N0, int N1, int N2){
//   real* result;
//   _fft_check_initialization();
//   result = (real*) fftwf_malloc(N0*N1*N2 * sizeof(real));
//   printf("fft_malloc_real\t(%d, %d, %d):\t%p\n", N0, N1, N2, result);
//   return result;
    return 0;
}


void fft_free(void* data){
//   printf("fft_free\t(%p)\n", data);
//   fftwf_free(data);
}


void* fft_init_forward(int N0, int N1, int N2, real* source, real* dest){
    cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
    
    cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C);
    plan->source = source;
    plan->dest = dest;
    cudaMalloc((void**)&plan->device, (2*N0*N1*N2) * sizeof(float));
    plan->N0 = N0;
    plan->N1 = N1;
    plan->N2 = N2;
    plan->direction = CUFFT_FORWARD;
    printf("fft_init_forward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
    return plan;
}


void* fft_init_backward(int N0, int N1, int N2, real* source, real* dest){
//     void* result = 
//     
//     printf("fft_init_backward\t(%d, %d, %d):\t%p\n", N0, N1, N2, result);
//     return result;
}


void fft_destroy_plan(void* plan){
  
}


void fft_execute(void* plan_ptr){
  int i;
  cudaPlan* plan = (cudaPlan*)plan_ptr;
  int N = plan->N0 * plan->N1 * plan->N2;
  
  if(plan->direction == CUFFT_FORWARD){
  for(i=0; i<N; i++){
    plan->device[2*i] = plan->source[i];
  }
  cufftExecC2C(plan->handle, (cufftComplex*)plan->device, (cufftComplex*)plan->device, plan->direction);
  }
  else if (plan->direction == CUFFT_INVERSE){
    
  }
  else{
    exit(3);
  }
    
  
}
