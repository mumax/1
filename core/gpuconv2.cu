#include "gpuconv2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/** For debugging: gets tensor from the GPU and prints to screen */
// void extract(const char* msg, float* data, int* size){
//   int N0 = size[0];
//   int N1 = size[1];
//   int N2 = size[2];
//   
//   printf("%s(%d x %d x %d){\n", msg, N0, N1, N2);
//   tensor* t = new_tensor(3, N0, N1, N2);
//   memcpy_from_gpu(data, t->list, tensor_length(t));
//   format_tensor(t, stdout);
//   printf("}\n\n");
// }

//_____________________________________________________________________________________________ convolution

void gpuconv2_exec(gpuconv2* conv, float* m, float* h){
  // set m and h components
  float* m_comp[3];
  float* h_comp[3];
  for(int i=0; i<3; i++){ 			// slice the contiguous m array in 3 equal pieces, one for each component
    m_comp[i] = &(m[i * conv->len_m_comp]); 
    h_comp[i] = &(h[i * conv->len_h_comp]); 
  }
  
  /*
  gpu_zero(conv->ft_h, conv->len_ft_h);							// zero-out field (h) components
  for(int i=0; i<3; i++){								// transform and convolve per magnetization component m_i
    gpu_zero(conv->ft_m_i, conv->len_ft_m_i);						// zero-out the padded magnetization buffer first
    gpu_copy_pad_r2c(conv->m_comp[i], conv->ft_m_i, conv->size[0], conv->size[1], conv->size[2]);	//copy mi into the padded magnetization buffer, converting to complex format
    //extract("ft_m_i", conv->ft_m_i, conv->paddedComplexSize);
		
		
    gpu_plan3d_real_input_exec(conv->fftplan, conv->ft_m_i, CUFFT_FORWARD);
    //extract("ft_m_i (transformed)", conv->ft_m_i, conv->paddedComplexSize);
    // TODO: asynchronous execution hazard !!
		
		
    for(int j=0; j<3; j++){								// apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
      gpu_kernel_mul(conv->ft_m_i, conv->ft_kernel[i][j], conv->ft_h_comp[j], conv->len_ft_m_i);
      //extract("ft_h_j", conv->ft_h_comp[j], conv->paddedComplexSize);
    }
  }
//  TODO: Save memory by performing gpu_kernel_mul + FFT_inverse + copy_unpad for each component in serie, memory savings are 2*paddedStorageN
  
  for(int i=0; i<3; i++){
    gpu_plan3d_real_input_exec(conv->fftplan, conv->ft_h_comp[i], CUFFT_INVERSE);		// Inplace backtransform of each of the padded h[i]-buffers
    gpu_copy_unpad_c2r(conv->ft_h_comp[i], conv->h_comp[i], conv->size[0], conv->size[1], conv->size[2]);
  }*/
}

__global__ void _gpu_kernel_mul2(float* ft_m_i, float* ft_kernel_ij, float* ft_h_j){
  /*int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  float rea = ft_m_i[e];
  float reb = ft_kernel_ij[e];
  float ima = ft_m_i[e + 1];
  float imb = ft_kernel_ij[e + 1];
  ft_h_j[e] 	+=  rea*reb - ima*imb;
  ft_h_j[e + 1] +=  rea*imb + ima*reb;
  */  
}


void gpu_kernel_mul2(float* ft_m_i, float* ft_kernel_ij, float* ft_h_comp_j, int nRealNumbers){
/*  timer_start("gpuconv2_kernel_mul");
//   assert(nRealNumbers > 0);
//   assert(nRealNumbers % 2 == 0);
//   int threadsPerBlock = 512;
//   int blocks = (nRealNumbers/2) / threadsPerBlock;
//   gpu_checkconf_int(blocks, threadsPerBlock);
//   _gpu_kernel_mul<<<blocks, threadsPerBlock>>>(ft_m_i, ft_kernel_ij, ft_h_comp_j);
//   cudaThreadSynchronize();
  timer_stop("gpuconv2_kernel_mul");*/
}

//_____________________________________________________________________________________________ kernel

void gpuconv2_checksize_kernel(gpuconv2* conv, tensor* kernel){
  // kernel should be rank 5 tensor with size 3 x 3 x 2*N0 x 2xN1 x 2xN2 (could be reduced a bit)
//   assert(kernel->rank == 5);
//   assert(kernel->size[0] == 3);
//   assert(kernel->size[1] == 3);
//   for(int i=0; i<3; i++){ assert(kernel->size[i+2] == 2 * conv->size[i]); }
//   
//   assert(kernel->size[2] == conv->paddedSize[0]);
//   assert(kernel->size[3] == conv->paddedSize[1]);
//   assert(kernel->size[4] == conv->paddedSize[2]);
}

void gpuconv2_alloc_ft_kernel(gpuconv2* conv){
  conv->ft_kernel = (float***)calloc(3, sizeof(float**));
  for(int i=0; i<3; i++){ 
    conv->ft_kernel[i] = (float**)calloc(3, sizeof(float*));
    for(int j=0; j<3; j++){
      conv->ft_kernel[i][j] = new_gpu_array(conv->len_ft_kernel_ij);
    }
  }
}

int* NO_ZERO_PAD = (int*)calloc(3, sizeof(int));

void gpuconv2_loadkernel(gpuconv2* conv, tensor* kernel){
  fprintf(stderr, "loadkernel %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpuconv2_checksize_kernel(conv, kernel);
  gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(kernel->size[2], kernel->size[3], kernel->size[4], NO_ZERO_PAD);
  float norm = 1.0/float(conv->fftplan->paddedN);
  float* complex_kernel_ij = new_ram_array(conv->len_ft_kernel_ij);
  for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
	
	/// @todo !!!!!!!!!!!!!!!!!!!!!!! 
	//memcpy_r2c(tensor_get(kernel, 5, i, j, 0, 0, 0), complex_kernel_ij, conv->len_kernel_ij);
	
	//normalize
	for(int e=0; e<conv->len_ft_kernel_ij; e++){
	  complex_kernel_ij[e] *= norm;
	}
	memcpy_to_gpu(complex_kernel_ij, conv->ft_kernel[i][j], conv->len_ft_kernel_ij);
	//extract("kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
	gpu_plan3d_real_input_forward(plan, conv->ft_kernel[i][j]);
	//extract("ft_kernel_ij", conv->ft_kernel[i][j], conv->paddedComplexSize);
    }
  }
  free(complex_kernel_ij);
  delete_gpu_plan3d_real_input(plan);
}

//_____________________________________________________________________________________________ new

// void gpuconv2_init_sizes(gpuconv2* conv, int N0, int N1, int N2, int* zero_pad){
//   
// }

void gpuconv2_init_kernel(gpuconv2* conv, tensor* kernel){
  conv->len_kernel_ij = conv->fftplan->paddedN;		//the length of each kernel component K[i][j] (eg: Kxy)
  conv->len_ft_kernel_ij = conv->fftplan->paddedStorageN;	//the length of each FFT'ed kernel component ~K[i][j] (eg: ~Kxy)
  gpuconv2_alloc_ft_kernel(conv);
  gpuconv2_loadkernel(conv, kernel);
}

void gpuconv2_init_m(gpuconv2* conv){
  conv->len_m = 3 * conv->fftplan->N;
  conv->len_m_comp = conv->fftplan->N; 
  conv->ft_m_i = new_gpu_array(conv->len_ft_m_i);
}

void gpuconv2_init_h(gpuconv2* conv){
  conv->len_h = conv->len_m;
  //conv->h = new_gpu_array(conv->len_h);
  conv->len_h_comp = conv->len_m_comp; 
//  conv->h_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){ 
//     conv->h_comp[i] = &(conv->h[i * conv->len_h_comp]); // slice the contiguous h array in 3 equal pieces, one for each component
//   }
  
  conv->len_ft_h = 3 * conv->len_ft_m_i;
  conv->ft_h = new_gpu_array(conv->len_ft_h);
  
  conv->len_ft_h_comp = conv->len_ft_m_i;
  conv->ft_h_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){ 
    conv->ft_h_comp[i] = &(conv->ft_h[i * conv->len_ft_h_comp]); // slice the contiguous ft_h array in 3 equal pieces, one for each component
  }
  
}

//_____________________________________________________________________________________________ new gpuconv2

gpuconv2* new_gpuconv2(int N0, int N1, int N2, tensor* kernel, int* zero_pad){
  gpuconv2* conv = (gpuconv2*)malloc(sizeof(gpuconv2));
  conv->fftplan = new_gpu_plan3d_real_input(N0, N1, N2, zero_pad);
  //gpuconv2_init_sizes(conv, N0, N1, N2);
  gpuconv2_init_kernel(conv, kernel);
  gpuconv2_init_m(conv);
  gpuconv2_init_h(conv);
  return conv;
}


//_____________________________________________________________________________________________ FFT

gpu_plan3d_real_input* new_gpu_plan3d_real_input(int N0, int N1, int N2, int* zero_pad){

  gpu_plan3d_real_input* plan = (gpu_plan3d_real_input*)malloc(sizeof(gpu_plan3d_real_input));
  
  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  plan->paddedStorageSize = (int*)calloc(3, sizeof(int));
    
  int* size = plan->size;
  int* paddedSize = plan->paddedSize;
  int* paddedStorageSize = plan->paddedStorageSize;
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
  plan->N = N0 * N1 * N2;
  
 
  plan->paddedSize[X] = (1 + zero_pad[X]) * N0; 
  plan->paddedSize[Y] = (1 + zero_pad[Y]) * N1; 
  plan->paddedSize[Z] = (1 + zero_pad[Z]) * N2;
  plan->paddedN = plan->paddedSize[0] * plan->paddedSize[1] * plan->paddedSize[2];
  
  plan->paddedStorageSize[X] = plan->paddedSize[X];
  plan->paddedStorageSize[Y] = plan->paddedSize[Y];
//  plan->paddedStorageSize[Z] = plan->paddedSize[Z] +  gpu_stride_float();   ///@todo aanpassen!!
  plan->paddedStorageSize[Z] = plan->paddedSize[Z] +  2;
  plan->paddedStorageN = paddedStorageSize[X] * paddedStorageSize[Y] * paddedStorageSize[Z];
  
  gpu_safe( cufftPlan1d(&(plan->fwPlanZ), plan->paddedSize[Z], CUFFT_R2C, 1) );
  gpu_safe( cufftPlan1d(&(plan->planY), plan->paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X]) );
  gpu_safe( cufftPlan1d(&(plan->planX), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y]) );
  gpu_safe( cufftPlan1d(&(plan->invPlanZ), plan->paddedSize[Z], CUFFT_C2R, 1) );
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}

//_____________________________________________________________________________________________ transpose

__global__ void _gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
    // N0 <-> N2
    // i  <-> k
    int N3 = 2;
    
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
    
    dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
    dest[k*N1*N0*N3 + j*N0*N3 + i*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
}

void gpu_transposeXZ_complex(float* source, float* dest, int N0, int N1, int N2){
  timer_start("transposeXZ"); /// @todo section is double-timed with FFT exec
  
  assert(source != dest); // must be out-of-place
  
  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;
  //int N3 = 2;
  
  dim3 gridsize(N0, N1, 1);	///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeXZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
  
  timer_stop("transposeXZ");
}

//_____________________________________________________________________________________________

__global__ void _gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
    // N1 <-> N2
    // j  <-> k
    
    int N3 = 2;

		int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

// 		int index_dest = i*N2*N1*N3 + k*N1*N3 + j*N3;
// 		int index_source = i*N1*N2*N3 + j*N2*N3 + k*N3;

		
    dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 0] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 0];
    dest[i*N2*N1*N3 + k*N1*N3 + j*N3 + 1] = source[i*N1*N2*N3 + j*N2*N3 + k*N3 + 1];
/*    dest[index_dest + 0] = source[index_source + 0];
    dest[index_dest + 1] = source[index_source + 1];*/
}

void gpu_transposeYZ_complex(float* source, float* dest, int N0, int N1, int N2){
  timer_start("transposeYZ");
  
  assert(source != dest); // must be out-of-place
  
  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;
  //int N3 = 2;
  
  dim3 gridsize(N0, N1, 1);	///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeYZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();
  
  timer_stop("transposeYZ");
}

//_____________________________________________________________________________________________ exec plan

void gpu_plan3d_real_input_forward(gpu_plan3d_real_input* plan, float* data){
  timer_start("gpu_plan3d_real_input_forward_exec");

  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2
  
  for(int i=0; i<size[X]; i++){
    for(int j=0; j<size[Y]; j++){
      float* row = &(data[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      gpu_safe( cufftExecR2C(plan->fwPlanZ, (cufftReal*)row,  (cufftComplex*)row) ); // all stays in data
    }
  }
  cudaThreadSynchronize();
  
  gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);					// it's now in data2
  gpu_safe( cufftExecC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data2, CUFFT_FORWARD) ); // it's now again in data
  cudaThreadSynchronize();
  
  gpu_transposeXZ_complex(data2, data, N0, N2, N1*N3); // size has changed due to previous transpose! // it's now in data2
  gpu_safe( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_FORWARD) ); // it's now again in data
  cudaThreadSynchronize();
  
  timer_stop("gpu_plan3d_real_input_forward_exec");
}

void gpu_plan3d_real_input_inverse(gpu_plan3d_real_input* plan, float* data){
  timer_start("gpu_plan3d_real_input_inverse_exec");

  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2

	// input data is XZ transpozed and stored in data, FFTs on X-arrays out of place towards data2
  gpu_safe( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) ); // it's now in data2
  cudaThreadSynchronize();
//  gpu_transposeXZ_complex(data2, data, N0, N2, N1*N3); // size has changed due to previous transpose! // it's now in data
  gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3); // size has changed due to previous transpose! // it's now in data
  
  gpu_safe( cufftExecC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) ); // it's now again in data2
  cudaThreadSynchronize();
//  gpu_transposeYZ_complex(data2, data, N0, N1, N2*N3);					// it's now in data
  gpu_transposeYZ_complex(data2, data, N0, N2, N1*N3);					// it's now in data

  for(int i=0; i<size[X]; i++){
    for(int j=0; j<size[Y]; j++){
      float* row = &(data[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      gpu_safe( cufftExecC2R(plan->invPlanZ, (cufftComplex*)row, (cufftReal*)row) ); // all stays in data
    }
  }
  cudaThreadSynchronize();
  
  timer_stop("gpu_plan3d_real_input_inverse_exec");
}

void delete_gpu_plan3d_real_input(gpu_plan3d_real_input* plan){
  
}

#ifdef __cplusplus
}
#endif