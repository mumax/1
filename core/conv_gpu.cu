#include "conv_gpu.h"
#include "tensor.h"
#include <cufft.h>
#include <stdio.h>
#include <assert.h>

typedef struct{
  cufftHandle handle;
}cuda_c2c_plan;

cuda_c2c_plan* gpu_init_c2c(int* size){
  cuda_c2c_plan* plan = (cuda_c2c_plan*) malloc(sizeof(cuda_c2c_plan));
  cufftPlan3d(&(plan->handle), size[0], size[1], size[2], CUFFT_C2C);
  return plan;
}

void gpu_exec_c2c(void* plan, tensor* data){
  
}

void conv_execute(convplan* p, tensor* m, tensor* h){
  assert(m->rank == 4);
  assert(h->rank == 4);
  
  tensor* ft_h = p->ft_h;
  tensor* ft_m_i = p->ft_m_i;
  int* size = p->size;	// note: m->size == {3, N0, N1, N2}, size = {N0, N1, N2};
  
  // Zero-out field components
  for(int i = 0; i < tensor_length(ft_h); i++){
    ft_h->list[i] = 0.;
  }
  
  // transform and convolve per magnetization component m_i
  for(int i = 0; i < 3; i++){
    
    // zero-out the padded magnetization buffer first
    for(int j = 0; j < tensor_length(ft_m_i); j++){
      ft_m_i->list[j] = 0.;
    }
    
     //copy the current magnetization component into the padded magnetization buffer
     // we convert real to complex format
     for(int i_= 0; i_< size[0]; i_++){
      for(int j_= 0; j_< size[1]; j_++){
	for(int k_= 0; k_< size[2]; k_++){
	  *tensor_get(ft_m_i, 3, i_, j_, 2 * k_) = *tensor_get(m, 4, i, i_, j_, k_);
	}
      }
     }
     
     format_tensor(ft_m_i, stdout);

  }
//     // then copy the current magnetization component into the padded magnetization buffer
//     CopyInto(m.Component(i).Array(), p.ft_m_i.Array());
// 
//     // in-place FFT of the padded magnetization
//     p.forward.Execute();
//    
//     // apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
//     ft_m_i := p.ft_m_i.List();
//     for j:=0; j<3; j++{
//       ft_h_j := p.ft_h[j].List();
//       for e:=0; e<len(ft_m_i); e+=2{
// 	rea := ft_m_i[e];
// 	reb := p.ft_kernel[i][j].List()[e];
// 	ima := ft_m_i[e+1];
// 	imb := p.ft_kernel[i][j].List()[e+1];
// 	ft_h_j[e] +=  rea*reb - ima*imb;
// 	ft_h_j[e+1] +=  rea*imb + ima*reb;
//       }
//     }

  
}


convplan* new_convplan(int* size, tensor* kernel){
  convplan* plan = (convplan*) malloc(sizeof(convplan));
  
  plan->size[0] = size[0];
  plan->size[1] = size[1];
  plan->size[2] = size[2];
  plan->paddedComplexSize[0] = size[0];
  plan->paddedComplexSize[1] = size[1];
  plan->paddedComplexSize[2] = 2*size[2];
  
  plan->ft_m_i = new_tensorN(3, plan->paddedComplexSize);
  plan->ft_h = new_tensor(4, 3, plan->paddedComplexSize[0], plan->paddedComplexSize[1], plan->paddedComplexSize[2]);
  
  return plan;
}


void delete_convplan(convplan* plan){
  
  free(plan);
}


// typedef struct{
//   cufftHandle handle;
//   
//   float* source;
//   float* transf;
//   
//   float* device;
//   float* device2;
//   
//   int N0, N1, N2;
//   int direction;
// } cudaPlan;
// 
// 
// 
// void* fft_init_forward(int N0, int N1, int N2, float* source, float* transf){
//     cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
//     
//     cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_R2C);
//     plan->source = source;
//     plan->transf = transf;
//     cudaMalloc((void**)&plan->device, (N0*N1*(N2+2)) * sizeof(float));
//     cudaMalloc((void**)&plan->device2, (N0*N1*(N2+2)) * sizeof(float));
//     plan->N0 = N0;
//     plan->N1 = N1;
//     plan->N2 = N2;
//     plan->direction = CUFFT_FORWARD;
//     printf("fft_init_forward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
//     return plan;
// }
// 
// 
// void* fft_init_backward(int N0, int N1, int N2, float* transf, float* source){
//     cudaPlan* plan = (cudaPlan*) malloc(sizeof(cudaPlan));
//     
//     cufftPlan3d(&(plan->handle), N0, N1, N2, CUFFT_C2R);
//     plan->source = source;
//     plan->transf = transf;
//     cudaMalloc((void**)&plan->device, (N0*N1*(N2+2)) * sizeof(float));
//     cudaMalloc((void**)&plan->device2, (N0*N1*(N2+2)) * sizeof(float));
//     plan->N0 = N0;
//     plan->N1 = N1;
//     plan->N2 = N2;
//     plan->direction = CUFFT_INVERSE;
//     printf("fft_init_backward\t(%d, %d, %d):\t%p\n", N0, N1, N2, plan);
//     return plan;
// }
// 
// 
// void fft_execute(void* plan_ptr){
//   
//   printf("fft_execute():\t%p\n", plan_ptr);
//   
//   cudaPlan* plan = (cudaPlan*)plan_ptr;
//   int N0 = plan->N0, N1 = plan->N1, N2 = plan->N2;
//   //int N = plan->N0 * plan->N1 * plan->N2;
//   printf("%d x %d x %d\n", N0, N1, N2);
//   
//   ///////////////////////////////////////////// forward ///////////////////////////////////////////
//   
//   if(plan->direction == CUFFT_FORWARD){
//     printf("fft_execute() [forward]:\t%p\n", plan_ptr);
//     
//     //printf("**ORIGINAL DATA:\n");
//     //format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);
//     
//     printf("memcpy: %d\n", cudaMemcpy(plan->device, plan->source, (N0*N1*N2) * sizeof(float), cudaMemcpyHostToDevice));
//     
//     // r2c transform
//     printf("r2c: %d\n",cufftExecR2C(plan->handle, (cufftReal*)plan->device, (cufftComplex*)plan->device2));
//     
//     // copy everything back
//     cudaMemcpy(plan->transf, plan->device2, N0*N1*(N2+2) * sizeof(float), cudaMemcpyDeviceToHost);
//     
//     //printf("**TRANSFORMED:\n");
//     //format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout);
//   }
//   
//   ////////////////////////////////////////// backward ///////////////////////////////////////////////
//   
//   else if (plan->direction == CUFFT_INVERSE){
//     printf("fft_execute() [backward]:\t%p\n", plan_ptr);
//     
//     //printf("**BACKTRANSF INPUT (HALF):\n"); 
//     //format_tensor(as_tensor(plan->transf, 3, N0, N1, N2+2), stdout); 
//     
//     printf("memcpy: %d\n", cudaMemcpy(plan->device, plan->transf, N0*N1*(N2+2) * sizeof(float), cudaMemcpyHostToDevice));
//     
//     cufftExecC2R(plan->handle, (cufftComplex*)plan->device, (cufftReal*)plan->device2);
//     
//     printf("memcpy: %d\n",cudaMemcpy(plan->source, plan->device2, (N0*N1*N2) * sizeof(float), cudaMemcpyDeviceToHost));
//     
//     //printf("**BACKTRANSF (HALF):\n"); 
//     //format_tensor(as_tensor(plan->source, 3, N0, N1, N2), stdout);
// 
//   }
//   /////////////////////////////////// not backward nor forward ///////////////////////////
//   
//   else{
//     printf("fft_execute() [illegal plan]:\t%p", plan_ptr);
//     exit(3);
//   }
// }
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// void fft_init(void){
// }
// 
// 
// void fft_finalize(void){
// }
// 
// float* fft_malloc(int N0, int N1, int N2){
//     return 0;
// }
// 
// 
// void fft_free(void* data){
// }
// 
// void fft_destroy_plan(void* plan){
// }
