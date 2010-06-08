#include "gpufft2.h"
#include "gpufft.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @todo on compute capability < 2.0, the first step is done serially...
 */
gpuFFT3dPlan* new_gpuFFT3dPlan(int* size, int* kernelsize){
  
  int N0 = size[X]; int N1 = size[Y]; int N2 = size[Z];
  assert(N0 > 1);   assert(N1 > 1);   assert(N2 > 1);
  
  gpuFFT3dPlan* plan = (gpuFFT3dPlan*)malloc(sizeof(gpuFFT3dPlan));
  
  plan->size = (int*)calloc(3, sizeof(int));    ///@todo not int* but int[3]
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  plan->paddedStorageSize = (int*)calloc(3, sizeof(int));
  
  int* paddedSize = plan->paddedSize;
  int* paddedStorageSize = plan->paddedStorageSize;
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
  plan->N = N0 * N1 * N2;
  
  plan->paddedSize[X] = kernelsize[X];
  plan->paddedSize[Y] = kernelsize[Y];
  plan->paddedSize[Z] = kernelsize[Z];
  plan->paddedN = plan->paddedSize[0] * plan->paddedSize[1] * plan->paddedSize[2];
  
  plan->paddedStorageSize[X] = plan->paddedSize[X];
  plan->paddedStorageSize[Y] = plan->paddedSize[Y];
  plan->paddedStorageSize[Z] = gpu_pad_to_stride( plan->paddedSize[Z] +  2 );
  plan->paddedStorageN = paddedStorageSize[X] * paddedStorageSize[Y] * paddedStorageSize[Z];
  
  ///@todo check these sizes !
  gpu_safe( cufftPlan1d(&(plan->fwPlanZ), plan->paddedSize[Z], CUFFT_R2C, 1) );
  gpu_safe( cufftPlan1d(&(plan->planY), plan->paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X] / 2) );          // IMPORTANT: the /2 is necessary because the complex transforms have only half the amount of elements (the elements are now complex numbers)
  gpu_safe( cufftPlan1d(&(plan->planX), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2) );
  gpu_safe( cufftPlan1d(&(plan->invPlanZ), plan->paddedSize[Z], CUFFT_C2R, 1) );
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}

///@todo seems to write out of bounds, inverse does not
void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, tensor* input, tensor* output){
  timer_start("gpu_plan3d_real_input_forward_exec");

  assert(input == output); ///@todo works only in-place for now
  assert(input->rank == 3);
  assert(output->rank == 3);
  for(int i=0; i<3; i++){
    assert( input->size[i] == plan->paddedStorageSize[i]);
    assert(output->size[i] == plan->paddedStorageSize[i]);
  }
  
  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  float* data = input->list;
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2
  
  for(int i=0; i<size[X]; i++){
    for(int j=0; j<size[Y]; j++){
      float* rowIn  = &( input->list[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      float* rowOut = &(output->list[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      gpu_safe( cufftExecR2C(plan->fwPlanZ, (cufftReal*)rowIn,  (cufftComplex*)rowIn) ); ///@todo change for out-of-place
    }
  }
  cudaThreadSynchronize();
  
  gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);                  // it's now in data2
  
  gpu_safe( cufftExecC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data2, CUFFT_FORWARD) ); // it's now again in data
  cudaThreadSynchronize();
  
  gpu_transposeXZ_complex(data2, data, N0, N2, N1*N3); // size has changed due to previous transpose! // it's now in data2
  gpu_safe( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)output->list, CUFFT_FORWARD) ); // it's now again in data
  cudaThreadSynchronize();
  
  timer_stop("gpu_plan3d_real_input_forward_exec");
}



void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, tensor* input, tensor* output){
  timer_start("gpu_plan3d_real_input_inverse_exec");
  
  assert(input == output); ///@todo works only in-place for now
  assert(input->rank == 3);
  assert(output->rank == 3);
  for(int i=0; i<3; i++){
    assert( input->size[i] == plan->paddedStorageSize[i]);
    assert(output->size[i] == plan->paddedStorageSize[i]);
  }
  
  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  float* data = input->list;
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2

 // input data is XZ transpozed and stored in data, FFTs on X-arrays out of place towards data2
  gpu_safe( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) ); // it's now in data2
  cudaThreadSynchronize();
//  gpu_transposeXZ_complex(data2, data, N0, N2, N1*N3); // size has changed due to previous transpose! // it's now in data
  gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3); // size has changed due to previous transpose! // it's now in data
  
  gpu_safe( cufftExecC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) ); // it's now again in data2
  cudaThreadSynchronize();
//  gpu_transposeYZ_complex(data2, data, N0, N1, N2*N3);                 // it's now in data
  gpu_transposeYZ_complex(data2, data, N0, N2, N1*N3);                   // it's now in data

  for(int i=0; i<size[X]; i++){
    for(int j=0; j<size[Y]; j++){
      float* rowIn  = &( input->list[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      float* rowOut = &(output->list[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
      gpu_safe( cufftExecC2R(plan->invPlanZ, (cufftComplex*)rowIn, (cufftReal*)rowOut) ); // all stays in data
    }
  }
  cudaThreadSynchronize();
  
  timer_stop("gpu_plan3d_real_input_inverse_exec");
}




//_____________________________________________________________________________________________ transpose

void gpu_tensor_transposeYZ_complex(tensor* source, tensor* dest){
  assert(source != dest);                       // must be out-of-place
  assert(source->rank == 3);
  assert(dest->rank == 3);
  assert(dest->size[Y] == source->size[Z]/2);   // interleaved complex format
  assert(dest->size[Z] == source->size[Y]*2);
  
  timer_start("transposeYZ");
  
  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  int N0 = source->size[X];
  int N1 = source->size[Y];
  int N2 = source->size[Z] / 2;
  int N3 = 2;
  
  dim3 gridsize(N0, N1, 1);  ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeYZ_complex<<<gridsize, blocksize>>>(source->list, dest->list, N0, N1, N2);
  cudaThreadSynchronize();
  
  timer_stop("transposeYZ");
}

void gpu_tensor_transposeXZ_complex(tensor* source, tensor* dest){
  assert(source != dest);                       // must be out-of-place
  assert(source->rank == 3);
  assert(dest->rank == 3);
  assert(dest->size[X] == source->size[Z]/2);   // interleaved complex format
  assert(dest->size[Z] == source->size[X]*2);
  
  timer_start("transposeXZ");
  
  // we treat the complex array as a N0 x N1 x N2 x 2 real array
  // after transposing it becomes N0 x N2 x N1 x 2
  int N0 = source->size[X];
  int N1 = source->size[Y];
  int N2 = source->size[Z] / 2;
  int N3 = 2;
  
  dim3 gridsize(N0, N1, 1);  ///@todo generalize!
  dim3 blocksize(N2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
  _gpu_transposeXZ_complex<<<gridsize, blocksize>>>(source->list, dest->list, N0, N1, N2);
  cudaThreadSynchronize();
  
  timer_stop("transposeXZ");
}


// //_____________________________________________________________________________________________ exec plan
// 
// void gpu_plan3d_real_input_forward(gpu_plan3d_real_input* plan, float* data){
//   timer_start("gpu_plan3d_real_input_forward_exec");
// 
//   int* size = plan->size;
//   int* pSSize = plan->paddedStorageSize;
//   int N0 = pSSize[X];
//   int N1 = pSSize[Y];
//   int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
//   int N3 = 2;
//   
//   float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2
//   
//   for(int i=0; i<size[X]; i++){
//     for(int j=0; j<size[Y]; j++){
//       float* row = &(data[i * pSSize[Y] * pSSize[Z] + j * pSSize[Z]]);
//       gpu_safe( cufftExecR2C(plan->fwPlanZ, (cufftReal*)row,  (cufftComplex*)row) ); // all stays in data
//     }
//   }
//   cudaThreadSynchronize();
//   
//   gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);					// it's now in data2
//   gpu_safe( cufftExecC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data2, CUFFT_FORWARD) ); // it's now again in data
//   cudaThreadSynchronize();
//   
//   gpu_transposeXZ_complex(data2, data, N0, N2, N1*N3); // size has changed due to previous transpose! // it's now in data2
//   gpu_safe( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_FORWARD) ); // it's now again in data
//   cudaThreadSynchronize();
//   
//   timer_stop("gpu_plan3d_real_input_forward_exec");
// }
// 
// void gpu_plan3d_real_input_inverse(gpu_plan3d_real_input* plan, float* data){
//   
// }
// 
// void delete_gpu_plan3d_real_input(gpu_plan3d_real_input* plan){
//   
// 	gpu_safe( cufftDestroy(plan->fwPlanZ) );
// 	gpu_safe( cufftDestroy(plan->invPlanZ) );
// 	gpu_safe( cufftDestroy(plan->planY) );
// 	gpu_safe( cufftDestroy(plan->planX) );
// 
// 	gpu_safe( cudaFree(plan->transp) ); 
// 	gpu_safe( cudaFree(plan->size) );
// 	gpu_safe( cudaFree(plan->paddedSize) );
// 	gpu_safe( cudaFree(plan->paddedStorageSize) );
// 	free(plan);
// 
// }


#ifdef __cplusplus
}
#endif