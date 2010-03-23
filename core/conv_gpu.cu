#include "conv_gpu.h"
#include "tensor.h"
#include <stdio.h>
#include <assert.h>

void conv_execute(convplan* p, tensor* m, tensor* h){
  assert(m->rank == 4);
  assert(h->rank == 4);
  
  // shorthand notations
  tensor* ft_h      = p->ft_h;
  tensor* ft_m_i    = p->ft_m_i;
  tensor* ft_kernel = p->ft_kernel;
  int*    size      = p->size;			// note: m->size == {3, N0, N1, N2}, size = {N0, N1, N2};
  
  // Zero-out field (h) components
  for(int i = 0; i < tensor_length(ft_h); i++){  ft_h->list[i] = 0.;  }
  
  // transform and convolve per magnetization component m_i
  for(int i = 0; i < 3; i++){
    
    // zero-out the padded magnetization buffer first
    for(int j = 0; j < tensor_length(ft_m_i); j++){  ft_m_i->list[j] = 0.;  }
    
     //copy the current magnetization component into the padded magnetization buffer
     // we convert real to complex format
     for(int i_= 0; i_< size[0]; i_++){
      for(int j_= 0; j_< size[1]; j_++){
	for(int k_= 0; k_< size[2]; k_++){
	  *tensor_get(ft_m_i, 3, i_, j_, 2 * k_) = *tensor_get(m, 4, i, i_, j_, k_);
	}
      }
     }
     //format_tensor(ft_m_i, stdout);
     gpu_exec_c2c(p->c2c_plan, ft_m_i, CUFFT_FORWARD);
     //format_tensor(ft_m_i, stdout);

     // apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
     for(int j=0; j<3; j++){
	float* ft_h_j = tensor_component(ft_h, j)->list;	// todo: clean
	for(int e=0; e<tensor_length(ft_m_i); e+=2){
	  float rea = ft_m_i->list[e];
	  float reb = tensor_component(tensor_component(ft_kernel, i), j)->list[e];
	  float ima = ft_m_i->list[e + 1];
	  float imb = tensor_component(tensor_component(ft_kernel, i), j)->list[e + 1];
	  ft_h_j[e] 	+=  rea*reb - ima*imb;
	  ft_h_j[e + 1] +=  rea*imb + ima*reb;
	}
     }
  }
  
  for(int i=0; i<3; i++){
    // Inplace backtransform of each of the padded H-buffers
    tensor* ft_h_i = tensor_component(ft_h, i);
    gpu_exec_c2c(p->c2c_plan, ft_h_i, CUFFT_INVERSE);
    // Copy region of interest (non-padding space) to destination
    for(int i_= 0; i_< size[0]; i_++){
      for(int j_= 0; j_< size[1]; j_++){
	for(int k_= 0; k_< size[2]; k_++){
	  *tensor_get(h, 4, i, i_, j_, k_) = *tensor_get(ft_h_i, 3, i_, j_, 2*k_);
	}
      }
     }
  }

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
  plan->ft_kernel = new_tensor(5, 3, 3, plan->paddedComplexSize[0], plan->paddedComplexSize[1], plan->paddedComplexSize[2]);
  plan->c2c_plan = gpu_init_c2c(size);
  
  _init_kernel(plan, kernel);
      
  return plan;
}


void _init_kernel(convplan* plan, tensor* kernel){
  tensor* ft_kernel = plan->ft_kernel;
  int* size = plan->size;
  float norm = size[0] * size[1] * size[2];
  
  for(int s=0; s<3; s++){
    for(int d=0; d<3; d++){
      
      for(int i_= 0; i_< size[0]; i_++){
	for(int j_= 0; j_< size[1]; j_++){
	  for(int k_= 0; k_< size[2]; k_++){
	    *tensor_get(ft_kernel, 5, s, d, i_, j_, 2 * k_) = *tensor_get(kernel, 5, s, d, i_, j_, k_) / norm;
	  }
	}
      }
      
      tensor* k_sd = tensor_component(tensor_component(ft_kernel, s), d);
      gpu_exec_c2c(plan->c2c_plan, k_sd, CUFFT_FORWARD);
      // todo: free tensor components.
    }
  }
}


void delete_convplan(convplan* plan){
  
  free(plan);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////// FFT


cuda_c2c_plan* gpu_init_c2c(int* size){
  cuda_c2c_plan* plan = (cuda_c2c_plan*) malloc(sizeof(cuda_c2c_plan));
  cufftPlan3d(&(plan->handle), size[0], size[1], size[2], CUFFT_C2C);
  
  //float* device_list;
  cudaMalloc((void**)&(plan->device_buffer), (size[0]*size[1]*size[2]) * 2*sizeof(float));
  //plan->device_data = as_tensor(device_list, 3, size[0], size[1], size[2]);
  return plan;
}


void gpu_exec_c2c(cuda_c2c_plan* plan, tensor* data, int direction){
  int N = tensor_length(data);
  printf("N=%d\n", N);
  printf("memcpy: %d\n", cudaMemcpy(plan->device_buffer, data->list, N*sizeof(float), cudaMemcpyHostToDevice));
  cufftExecC2C(plan->handle, (cufftComplex*)plan->device_buffer, (cufftComplex*)plan->device_buffer, direction);
  printf("memcpy: %d\n", cudaMemcpy(data->list, plan->device_buffer, N*sizeof(float), cudaMemcpyDeviceToHost));
}