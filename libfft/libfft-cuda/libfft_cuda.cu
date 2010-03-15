#include "../libfft.h"
#include "../../libtensor/libtensor.h"
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
    plan->device_buf = (float*)malloc((2*N0*N1*N2) * sizeof(float));
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
    plan->device_buf = (float*)malloc((2*N0*N1*N2) * sizeof(float));
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
  
  // this approach seems logically flawed but incidentally worked for the special input data I used in the first tests
  // anyways, it was only meant as a quick way to get r2c transforms on the GPU, it is now clear that a full
  // convolution should actually be simpler to implement, so there is no need to try to correct this code.
  
  printf("fft_execute():\t%p\n", plan_ptr);
  
  int i, j, k;
  cudaPlan* plan = (cudaPlan*)plan_ptr;
  int N0 = plan->N0, N1 = plan->N1, N2 = plan->N2;
  int N = plan->N0 * plan->N1 * plan->N2;
  printf("%d x %d x %d = %d\n", N0, N1, N2, N);
  
  ///////////////////////////////////////////// forward ///////////////////////////////////////////
  
  if(plan->direction == CUFFT_FORWARD){
    printf("fft_execute() [forward]:\t%p\n", plan_ptr);
    
    
    printf("**ORIGINAL DATA:\n");
    format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);
    
    
    // convert to real [a, b, c, ...] to complex [a, 0, b, 0, c, 0, ...]
    for(i=0; i<N; i++){
      plan->device_buf[2*i] = plan->source[i];
      plan->device_buf[2*i+1] = 0.0;
    }
    
    printf("**COMPLEX ORIGINAL DATA:\n");
    format_tensor(as_tensor(plan->device_buf, 3, N0, N1, 2*N2), stdout);
    
    // copy complex data [a, 0, b, 0, c, 0, ...] to GPU
    cudaMemcpy(plan->device, plan->device_buf, 2*N * sizeof(float), cudaMemcpyHostToDevice);
    
    // c2c transform
    cufftExecC2C(plan->handle, (cufftComplex*)plan->device, (cufftComplex*)plan->device, plan->direction);
    
    // copy everything back, full complex format
    cudaMemcpy(plan->device_buf, plan->device, 2*N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("**TRANSFORMED (FULL):\n"); // todo: check conjugacy
    format_tensor(as_tensor(plan->device_buf, 3, N0, N1, 2*N2), stdout);
    
    // copy only half(+1) of the transformed data back, the rest is conjugate. 
    for(i=0; i < N0; i++){
      for(j=0; j <  N1; j++){
	for(k=0; k < N2+2; k++){
	  plan->transf[ (i*N1+j)*(N2+2)+k ] = plan->device_buf[ (i*N1+j)*(2*N2)+k ]; 
	}
      }
    }
    
    printf("**TRANSFORMED (HALF):\n"); // todo: check conjugacy
    format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout);
    
  }
  
  ////////////////////////////////////////// backward ///////////////////////////////////////////////
  
  else if (plan->direction == CUFFT_INVERSE){
    printf("fft_execute() [backward]:\t%p\n", plan_ptr);
    
    printf("**BACKTRANSF INPUT (HALF):\n"); // todo: check conjugacy
    format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout);
    
    // convert the half-complex format into full complex: the second half is the conjugate of the first half
    for(i=0; i < N0; i++){
      for(j=0; j < N1; j++){
	// first half: just copy
	for(k=0; k < N2+2; k++){  // < N2+2
	  plan->device_buf[ (i*N1+j)*(2*N2)+k ] = plan->transf[ (i*N1+j)*(N2+2) +k ]; 
	}
	// second half: copy + conjugate: does not work
	for(k=N2+2; k < 2*N2; k+=2){
	  plan->device_buf[ (i*N1+j)*(2*N2) + k] = plan->transf[ (i*N1+j)*(N2+2) + 2*N2 - k ];
	  plan->device_buf[ (i*N1+j)*(2*N2) + k + 1] = - plan->transf[ (i*N1+j)*(N2+2) + 2*N2 - k + 1]; 
	}
      }
    }
    
    printf("**BACKTRANSF INPUT (FULL ):\n"); // todo: check conjugacy: OK?
    format_tensor(as_tensor(plan->device_buf, 3, N0, N1, 2*N2), stdout);
    
    cudaMemcpy(plan->device, plan->device_buf, 2*N * sizeof(float), cudaMemcpyHostToDevice);
    
    cufftExecC2C(plan->handle, (cufftComplex*)plan->device, (cufftComplex*)plan->device, plan->direction);
    
    cudaMemcpy(plan->device_buf, plan->device, 2*N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("**BACKTRANSF (FULL):\n"); // todo: check realness: KO: most imaginary parts nearly zero, but some are large!
				      
    format_tensor(as_tensor(plan->device_buf, 3, N0, N1, 2*N2), stdout);

    
    // copy only real part backward
    // todo: check that imag part is zero.
    for(i=0; i<N; i++){
      plan->source[i] = plan->device_buf[2*i];
    }
    
    printf("**BACKTRANSF (HALF):\n"); 
    format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);

  }
  /////////////////////////////////// not backward nor forward ///////////////////////////
  
  else{
    printf("fft_execute() [illegal plan]:\t%p", plan_ptr);
    exit(3);
  }
}
