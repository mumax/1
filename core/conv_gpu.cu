#include "conv_gpu.h"

#include <cufft.h>
#include <stdio.h>

typedef struct{
  cufftHandle handle;
  
  float* source;
  float* transf;
  
  float* device;
  float* device2;
  
  int N0, N1, N2;
  int direction;
} cudaPlan;



void* fft_init_forward(int N0, int N1, int N2, float* source, float* transf){
    cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
    
    cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_R2C);
    plan->source = source;
    plan->transf = transf;
    cudaMalloc((void**)&plan->device, (N0*N1*(N2+2)) * sizeof(float));
    cudaMalloc((void**)&plan->device2, (N0*N1*(N2+2)) * sizeof(float));
    plan->N0 = N0;
    plan->N1 = N1;
    plan->N2 = N2;
    plan->direction = CUFFT_FORWARD;
    printf("fft_init_forward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
    return plan;
}


void* fft_init_backward(int N0, int N1, int N2, float* transf, float* source){
    cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
    
    cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2R);
    plan->source = source;
    plan->transf = transf;
    cudaMalloc((void**)&plan->device, (N0*N1*(N2+2)) * sizeof(float));
    cudaMalloc((void**)&plan->device2, (N0*N1*(N2+2)) * sizeof(float));
    plan->N0 = N0;
    plan->N1 = N1;
    plan->N2 = N2;
    plan->direction = CUFFT_INVERSE;
    printf("fft_init_backward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
    return plan;
}


void fft_execute(void* plan_ptr){
  
  printf("fft_execute():\t%p\n", plan_ptr);
  
  cudaPlan* plan = (cudaPlan*)plan_ptr;
  int N0 = plan->N0, N1 = plan->N1, N2 = plan->N2;
  //int N = plan->N0 * plan->N1 * plan->N2;
  printf("%d x %d x %d\n", N0, N1, N2);
  
  ///////////////////////////////////////////// forward ///////////////////////////////////////////
  
  if(plan->direction == CUFFT_FORWARD){
    printf("fft_execute() [forward]:\t%p\n", plan_ptr);
    
    //printf("**ORIGINAL DATA:\n");
    //format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);
    
    printf("memcpy: %d\n", cudaMemcpy(plan->device, plan->source, (N0*N1*N2) * sizeof(float), cudaMemcpyHostToDevice));
    
    // r2c transform
    printf("r2c: %d\n",cufftExecR2C(plan->handle, (cufftReal*)plan->device, (cufftComplex*)plan->device2));
    
    // copy everything back
    cudaMemcpy(plan->transf, plan->device2, N0*N1*(N2+2) * sizeof(float), cudaMemcpyDeviceToHost);
    
    //printf("**TRANSFORMED:\n");
    //format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout);
  }
  
  ////////////////////////////////////////// backward ///////////////////////////////////////////////
  
  else if (plan->direction == CUFFT_INVERSE){
    printf("fft_execute() [backward]:\t%p\n", plan_ptr);
    
    //printf("**BACKTRANSF INPUT (HALF):\n"); 
    //format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout); 
    
    printf("memcpy: %d\n", cudaMemcpy(plan->device, plan->transf, N0*N1*(N2+2) * sizeof(float), cudaMemcpyHostToDevice));
    
    cufftExecC2R(plan->handle, (cufftComplex*)plan->device, (cufftReal*)plan->device2);
    
    printf("memcpy: %d\n",cudaMemcpy(plan->source, plan->device2, (N0*N1*N2) * sizeof(float), cudaMemcpyDeviceToHost));
    
    //printf("**BACKTRANSF (HALF):\n"); 
    //format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);

  }
  /////////////////////////////////// not backward nor forward ///////////////////////////
  
  else{
    printf("fft_execute() [illegal plan]:\t%p", plan_ptr);
    exit(3);
  }
}










void fft_init(void){
}


void fft_finalize(void){
}

float* fft_malloc(int N0, int N1, int N2){
    return 0;
}


void fft_free(void* data){
}

void fft_destroy_plan(void* plan){
}
