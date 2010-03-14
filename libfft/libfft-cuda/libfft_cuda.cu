#include "../libfft.h"
#include <cufft.h>

#include <stdio.h>

/** An FFTW plan "knows" everything it needs to know, including its source and transfination arrays and direction (forward/backward). We can thus just call execute(fftw_plan). A CUDA plan, on the other hand, does not store this information. Therefore, we wrap this information in a cudaPlan, which is similar to an FFTW_plan. This allows us to call execute(plan), regardless of the plan being an FFTW or CUDA plan.
*/
typedef struct{
  cufftHandle handle;
  
  real* source;
  real* transf;
  
  real* device;
  real* device_buf;
  
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


void* fft_init_forward(int N0, int N1, int N2, real* source, real* transf){
    cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
    
    cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C);
    plan->source = source;
    plan->transf = transf;
    cudaMalloc((void**)&plan->device, (2*N0*N1*N2) * sizeof(float));
    //plan->device = 
    plan->N0 = N0;
    plan->N1 = N1;
    plan->N2 = N2;
    plan->direction = CUFFT_FORWARD;
    printf("fft_init_forward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
    return plan;
}


void* fft_init_backward(int N0, int N1, int N2, real* transf, real* source){
    cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
    
    cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2C);
    plan->source = source;
    plan->transf = transf;
    cudaMalloc((void**)&plan->device, (2*N0*N1*N2) * sizeof(float));
    plan->N0 = N0;
    plan->N1 = N1;
    plan->N2 = N2;
    plan->direction = CUFFT_INVERSE;
    printf("fft_init_backward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
    return plan;
}


void fft_transfroy_plan(void* plan){
  
}


void fft_execute(void* plan_ptr){
  printf("fft_execute():\t%p\n", plan_ptr);
  int i, j, k;
  cudaPlan* plan = (cudaPlan*)plan_ptr;
  int N0 = plan->N0, N1 = plan->N1, N2 = plan->N2;
  int N = plan->N0 * plan->N1 * plan->N2;
  printf("%d x %d x %d = %d\n", N0, N1, N2, N);
  if(plan->direction == CUFFT_FORWARD){
    printf("fft_execute() [forward]:\t%p\n", plan_ptr);
    for(i=0; i<N; i++){
      plan->device[2*i] = plan->source[i]; // segfault...
    }
    
    cufftExecC2C(plan->handle, (cufftComplex*)plan->device, (cufftComplex*)plan->device, plan->direction);
    
    for(i=0; i < N0; i++){
      for(j=0; j <  N1; j++){
	for(k=0; k < N2+2; k++){
	  plan->transf[ (i*N1+j)*N2+k ] = plan->device[ (i*N1+j)*N2+k ]; 
	}
      }
    }
  }
  else if (plan->direction == CUFFT_INVERSE){
    printf("fft_execute() [backward]:\t%p\n", plan_ptr);
    for(i=0; i < N0; i++){
      for(j=0; j < N1; j++){
	for(k=0; k < N2+2; k++){
	  plan->device[ (i*N1+j)*N2+k ] = plan->transf[ (i*N1+j)*N2+k ]; 
	}
	for(k= N2+2; k < N2; k+=2){
	  plan->device[ (i*N1+j)*N2+k ] = plan->device[ (i*N1+j)*N2+k ];
	  plan->device[ (i*N1+j)*N2+k +1 ] = -plan->transf[ (i*N1+j)*N2+k +1 ]; 
	}
	
	cufftExecC2C(plan->handle, (cufftComplex*)plan->device, (cufftComplex*)plan->device, plan->direction);
	
	for(i=0; i<N; i++){
	   plan->source[i] = plan->device[2*i];
	}
      }
    }
  }
  else{
    printf("fft_execute() [illegal plan]:\t%p", plan_ptr);
    exit(3);
  }
}
