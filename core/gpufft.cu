#include "gpufft.h"
#include "gpuconv2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


gpu_plan3d_real_input* new_gpu_plan3d_real_input(int N0, int N1, int N2, int* zero_pad){
  assert(N0 > 1);
  assert(N1 > 1);
  assert(N2 > 1);
  
  gpu_plan3d_real_input* plan = (gpu_plan3d_real_input*) malloc(sizeof(gpu_plan3d_real_input));
  
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
  plan->paddedStorageSize[Z] = gpu_pad_to_stride( plan->paddedSize[Z] +  2 );
  plan->paddedStorageN = paddedStorageSize[X] * paddedStorageSize[Y] * paddedStorageSize[Z];
  
  gpu_safe( cufftPlan1d(&(plan->fwPlanZ), plan->paddedSize[Z], CUFFT_R2C, 1) );
  gpu_safe( cufftPlan1d(&(plan->planY), plan->paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X] / 2) ); // IMPORTANT: the /2 is necessary because the complex transforms have only half the amount of elements (the elements are now complex numbers)
  gpu_safe( cufftPlan1d(&(plan->planX), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2) );
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

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
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

//      int index_dest = i*N2*N1*N3 + k*N1*N3 + j*N3;
//      int index_source = i*N1*N2*N3 + j*N2*N3 + k*N3;


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

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeYZ_complex<<<gridsize, blocksize>>>(source, dest, N0, N1, N2);
  cudaThreadSynchronize();

  timer_stop("transposeYZ");
}

//____________________________________________________________________________________ inplace

__global__ void _gpu_transposeXZ_complex_inplace(float* source, int N0, int N1, int N2){
    // N0 <-> N2
    // i  <-> k
    int N3 = 2;

    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

    int reDestIndex   = k*N1*N0*N3 + j*N0*N3 + i*N3 + 0;    ///@todo there's a bit redundancy here, but let's keep it clear for now...
    int imDestIndex   = k*N1*N0*N3 + j*N0*N3 + i*N3 + 1;
    int reSourceIndex = i*N1*N2*N3 + j*N2*N3 + k*N3 + 0;
    int imSourceIndex = i*N1*N2*N3 + j*N2*N3 + k*N3 + 1;

    if(reDestIndex < reSourceIndex){  // this is a bit of a trick to make sure we only swap each pair once. >, >= or <= would work equally well
      float            temp = source[reSourceIndex];
      source[reSourceIndex] = source[reDestIndex];
      source[reDestIndex]   = temp;

                       temp = source[imSourceIndex];
      source[imSourceIndex] = source[imDestIndex];
      source[imDestIndex]   = temp;
    }
}

void gpu_transposeXZ_complex_inplace(float* source, int N0, int N1, int N2){
  timer_start("transposeXZ_inplace"); /// @todo section is double-timed with FFT exec

  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;
  //int N3 = 2;

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeXZ_complex_inplace<<<gridsize, blocksize>>>(source, N0, N1, N2);
  cudaThreadSynchronize();

  timer_stop("transposeXZ_inplace");
}

//_____________________________________________________________________________________________

__global__ void _gpu_transposeYZ_complex_inplace(float* source, int N0, int N1, int N2){
    // N1 <-> N2
    // j  <-> k

    int N3 = 2;

    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

    int reDestIndex   = i*N2*N1*N3 + k*N1*N3 + j*N3 + 0;    ///@todo there's a bit redundancy here, but let's keep it clear for now...
    int imDestIndex   = i*N2*N1*N3 + k*N1*N3 + j*N3 + 1;
    int reSourceIndex = i*N1*N2*N3 + j*N2*N3 + k*N3 + 0;
    int imSourceIndex = i*N1*N2*N3 + j*N2*N3 + k*N3 + 1;

    if(reDestIndex < reSourceIndex){  // this is a bit of a trick to make sure we only swap each pair once. >, >= or <= would work equally well
      float            temp = source[reSourceIndex];
      source[reSourceIndex] = source[reDestIndex];
      source[reDestIndex]   = temp;

                       temp = source[imSourceIndex];
      source[imSourceIndex] = source[imDestIndex];
      source[imDestIndex]   = temp;
    }
}

void gpu_transposeYZ_complex_inplace(float* source, int N0, int N1, int N2){
  timer_start("transposeYZ_inplace");

  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  N2 /= 2;
  //int N3 = 2;

  dim3 gridsize(N0, N1, 1); ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeYZ_complex_inplace<<<gridsize, blocksize>>>(source, N0, N1, N2);
  cudaThreadSynchronize();

  timer_stop("transposeYZ_inplace");
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
  
	gpu_safe( cufftDestroy(plan->fwPlanZ) );
	gpu_safe( cufftDestroy(plan->invPlanZ) );
	gpu_safe( cufftDestroy(plan->planY) );
	gpu_safe( cufftDestroy(plan->planX) );

	gpu_safe( cudaFree(plan->transp) ); 
	gpu_safe( cudaFree(plan->size) );
	gpu_safe( cudaFree(plan->paddedSize) );
	gpu_safe( cudaFree(plan->paddedStorageSize) );
	free(plan);

}


#ifdef __cplusplus
}
#endif