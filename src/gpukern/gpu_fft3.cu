#include "gpu_fft3.h"
<<<<<<< HEAD
#include "gpu_mem.h"
#include "gpu_transpose2.h"
#include "gpu_safe.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>
#include "../macros.h"
=======
#include "../macros.h"
#include "gpu_transpose.h"
#include "gpu_safe.h"
#include "gpu_conf.h"
#include "gpu_mem.h"
#include "gpu_zeropad.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>
>>>>>>> arne

#ifdef __cplusplus
extern "C" {
#endif

<<<<<<< HEAD


=======
/// DEBUG: print real block
void print(char* tag, float* data, int N0, int N1, int N2){
  printf("%s (%d x %d x %d)\n", tag, N0, N1, N2);
  for(int i=0; i<N0; i++){
    for(int j=0; j<N1; j++){
      for(int k=0; k<N2; k++){
        fprintf(stdout, "%g\t", gpu_array_get(data, i*N1*N2 + j*N2 + k));
      }
      printf("\n");
    }
    printf("\n");
  }
}

/// DEBUG: print complex block
void printc(char* tag, float* data, int N0, int N1, int N2, int N3){
  printf("%s (%d x %d x %d x %d)\n", tag, N0, N1, N2, N3);
  for(int i=0; i<N0; i++){
    for(int j=0; j<N1; j++){
      for(int k=0; k<N2; k++){
        for(int l=0; l<N3; l++){
          fprintf(stdout, "%g\t", gpu_array_get(data, i*N1*N2*N3 + j*N2*N3 + k*N3 + l));
        }
        printf("  ");
      }
      printf("\n");
    }
    printf("\n");
  }
}


/**
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @TODO interleave with streams: fft_z(MX), transpose(MX) async, FFT_z(MY), transpose(MY) async, ... threadsync, FFT_Y(MX)...
 * @TODO CUFFT crashes on more than one million cells, seems to be exactly the limit of 8 million elements (after zero-padding + 2 extra rows)
 */
>>>>>>> arne
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, int* paddedSize){
  
  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  gpuFFT3dPlan* plan = (gpuFFT3dPlan*)malloc(sizeof(gpuFFT3dPlan));
  
<<<<<<< HEAD
  plan->size = (int*)calloc(3, sizeof(int));    ///@todo not int* but int[3]
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  plan->paddedStorageSize = (int*)calloc(3, sizeof(int));
  
//   int* paddedSize = plan->paddedSize;
  int* paddedStorageSize = plan->paddedStorageSize;
=======
  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
>>>>>>> arne
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
<<<<<<< HEAD
  plan->N = N0 * N1 * N2;
=======
>>>>>>> arne
  
  plan->paddedSize[X] = paddedSize[X];
  plan->paddedSize[Y] = paddedSize[Y];
  plan->paddedSize[Z] = paddedSize[Z];
  plan->paddedN = plan->paddedSize[0] * plan->paddedSize[1] * plan->paddedSize[2];
<<<<<<< HEAD
  
  plan->paddedStorageSize[X] = plan->paddedSize[X];
  plan->paddedStorageSize[Y] = plan->paddedSize[Y];
  plan->paddedStorageSize[Z] = gpu_pad_to_stride( plan->paddedSize[Z] + 2 );
  plan->paddedStorageN = paddedStorageSize[X] * paddedStorageSize[Y] * paddedStorageSize[Z];
  
  gpu_safefft( cufftPlan1d(&(plan->fwPlanZ), plan->paddedSize[Z], CUFFT_R2C, size[X]*size[Y]) );
  gpu_safefft( cufftPlan1d(&(plan->planY), plan->paddedSize[Y], CUFFT_C2C, paddedStorageSize[Z] * size[X] / 2) );          // IMPORTANT: the /2 is necessary because the complex transforms have only half the amount of elements (the elements are now complex numbers)
  gpu_safefft( cufftPlan1d(&(plan->planX), plan->paddedSize[X], CUFFT_C2C, paddedStorageSize[Z] * paddedSize[Y] / 2) );
  gpu_safefft( cufftPlan1d(&(plan->invPlanZ), plan->paddedSize[Z], CUFFT_C2R, size[X]*size[Y]) );
  
  plan->transp = new_gpu_array(plan->paddedStorageN);
  
  return plan;
}






// void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, tensor* input, tensor* output){
//   assertDevice(input->list);
//   assertDevice(output->list);
//   assert(input->list == output->list); ///@todo works only in-place for now
//   assert(input->rank == 3);
//   assert(output->rank == 3);
//   for(int i=0; i<3; i++){
//     assert( input->size[i] == plan->paddedStorageSize[i]);
//     assert(output->size[i] == plan->paddedStorageSize[i]);
//   }
//   
//   gpuFFT3dPlan_forward_unsafe(plan, input->list, output->list);
// 
//   return;
// }

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, float* input, float* output){
  timer_start("gpu_plan3d_real_input_forward_exec");
  
  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  int half_pSSize = plan->paddedStorageN/2;
  
  float* data = input;
  float* data2 = plan->transp; 

  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      // out of place FFTs in Z-direction from the 0-element towards second half of the zeropadded matrix (out of place: no +2 on input!)
    gpu_safefft( cufftExecR2C(plan->fwPlanZ, (cufftReal*)data,  (cufftComplex*) (data + half_pSSize) ) );     // it's in data
    cudaThreadSynchronize();
      // zero out the input data points at the start of the matrix
    gpu_zero(data, size[X]*size[Y]*pSSize[Z]);
    
      // YZ-transpose within the same matrix from the second half of the matrix towards the 0-element
    yz_transpose_in_place_fw(data, size, pSSize);                                                          // it's in data
    
      // in place FFTs in Y-direction
    gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_FORWARD) );       // it's in data 
    cudaThreadSynchronize();
  }
  
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // in place FFTs in Z-direction (there is no zero space to perform them out of place)
    gpu_safefft( cufftExecR2C(plan->fwPlanZ, (cufftReal*)data,  (cufftComplex*) data ) );                     // it's in data
    cudaThreadSynchronize();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data, data2, N0, N1, N2*N3);                                                   // it's in data2
    
      // perform the FFTs in the Y-direction
    gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)data2,  (cufftComplex*)data, CUFFT_FORWARD) );      // it's in data
    cudaThreadSynchronize();
  }

  if(N0 > 1){    // not done for 2D transforms
      // XZ transpose still needs to be out of place
    gpu_transposeXZ_complex(data, data2, N0, N2, N1*N3);                                                   // it's in data2
    
      // in place FFTs in X-direction
    gpu_safefft( cufftExecC2C(plan->planX, (cufftComplex*)data2,  (cufftComplex*)output, CUFFT_FORWARD) );    // it's in data
    cudaThreadSynchronize();
  }

  timer_stop("gpu_plan3d_real_input_forward_exec");
  
  return;
}







// void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, tensor* input, tensor* output){
//   assertDevice(input->list);
//   assertDevice(output->list);
//   assert(input->list == output->list); ///@todo works only in-place for now
//   assert(input->rank == 3);
//   assert(output->rank == 3);
//   for(int i=0; i<3; i++){
//     assert( input->size[i] == plan->paddedStorageSize[i]);
//     assert(output->size[i] == plan->paddedStorageSize[i]);
//   }
//   gpuFFT3dPlan_inverse_unsafe(plan, input->list, output->list);
//   
//   return;
// }


void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, float* input, float* output){
  
  timer_start("gpu_plan3d_real_input_inverse_exec");
  
  int* size = plan->size;
  int* pSSize = plan->paddedStorageSize;
  int N0 = pSSize[X];
  int N1 = pSSize[Y];
  int N2 = pSSize[Z]/2; // we treat the complex data as an N0 x N1 x N2 x 2 array
  int N3 = 2;
  
  float* data = input;
  float* data2 = plan->transp; // both the transpose and FFT are out-of-place between data and data2

  if (N0 > 1){
      // out of place FFTs in the X-direction (i.e. no +2 stride on input!)
    gpu_safefft( cufftExecC2C(plan->planX, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) );      // it's in data2
    cudaThreadSynchronize();

      // XZ transpose still needs to be out of place
    gpu_transposeXZ_complex(data2, data, N1, N2, N0*N3);                                                   // it's in data
  }
  
  if ( pSSize[X]!=size[X] || pSSize[Y]!=size[Y]){
      // in place FFTs in Y-direction
    gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data, CUFFT_INVERSE) );        // it's in data
    cudaThreadSynchronize();

      // YZ-transpose within the same matrix from the 0-element towards the second half of the matrix
    yz_transpose_in_place_inv(data, size, pSSize);                                                          // it's in data

      // out of place FFTs in Z-direction from the second half of the matrix towards the 0-element
    gpu_safefft( cufftExecC2R(plan->invPlanZ, (cufftComplex*)(data + N0*N1*N2), (cufftReal*)data ));           // it's in data
    cudaThreadSynchronize();

  }
  else {          //no zero padding in X- and Y direction (e.g. for Greens kernel computations)
      // out of place FFTs in Y-direction
    gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)data,  (cufftComplex*)data2, CUFFT_INVERSE) );       // it's in data
    cudaThreadSynchronize();
    
      // YZ-transpose needs to be out of place.
    gpu_transposeYZ_complex(data2, data, N0, N2, N1*N3);                                                    // it's in data2   

      // in place FFTs in Z-direction
    gpu_safefft( cufftExecC2R(plan->invPlanZ, (cufftComplex*) data, (cufftReal*) data ));                      // it's in data
    cudaThreadSynchronize();
  }
 
  timer_stop("gpu_plan3d_real_input_inverse_exec");
  
  return;
=======

  int complexZ = (paddedSize[Z] + 2)/2;

  gpu_safefft( cufftPlan1d(&(plan->fwPlanZ),  plan->paddedSize[Z],   CUFFT_R2C, size[X] * size[Y]) );
  gpu_safefft( cufftPlan1d(&(plan->invPlanZ), plan->paddedSize[Z],   CUFFT_C2R, size[X] * size[Y]) );
  gpu_safefft( cufftPlan1d(&(plan->planY),    plan->paddedSize[Y], CUFFT_C2C, complexZ * size[X]) );
  gpu_safefft( cufftPlan1d(&(plan->planX),    plan->paddedSize[X], CUFFT_C2C, complexZ * paddedSize[Y]) );

  gpu_safefft( cufftSetCompatibilityMode(plan->fwPlanZ, CUFFT_COMPATIBILITY_NATIVE) );
  gpu_safefft( cufftSetCompatibilityMode(plan->invPlanZ, CUFFT_COMPATIBILITY_NATIVE) );
  gpu_safefft( cufftSetCompatibilityMode(plan->planY, CUFFT_COMPATIBILITY_NATIVE) );
  gpu_safefft( cufftSetCompatibilityMode(plan->planX, CUFFT_COMPATIBILITY_NATIVE) );


  // IMPORTANT: the /2 is necessary because the complex transforms have only half the amount of elements (the elements are now complex numbers)
  
  
  plan->buffer1  = new_gpu_array(size[X] * size[Y] * paddedSize[Z]);       // padding in Z direction
  plan->buffer2  = new_gpu_array(size[X] * size[Y] * complexZ * 2);        // half complex
  plan->buffer2t = new_gpu_array(size[X] * complexZ * size[Y] * 2);        // transposed @todo: get rid of: combine transpose+pad in one operation
  plan->buffer3  = new_gpu_array(size[X] * complexZ * paddedSize[Y] * 2);  //transposed and padded
  plan->buffer3t = new_gpu_array(paddedSize[Y] * complexZ * size[X] * 2);  //transposed @todo: get rid of: combine transpose+pad in one operation
  //output size  :               paddedSize[Y] * paddedComplexSize[Z] * paddedSize[X]
  
  return plan;
}


gpuFFT3dPlan* new_gpuFFT3dPlan(int* size){
  return new_gpuFFT3dPlan_padded(size, size); // when size == paddedsize, there is no padding
}

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, float* input, float* output){

  // complex numbers in the (original) Z direction after the R2C transform
  int complexZ = (plan->paddedSize[Z] + 2)/2;
  
  int* size = plan->size;
  int* paddedSize = plan->paddedSize;
  float* buffer1 = plan->buffer1;
  float* buffer2 = plan->buffer2;
  float* buffer2t = plan->buffer2t;
  float* buffer3 = plan->buffer3;
  float* buffer3t = plan->buffer3t;

  //print("input", input, size[X], size[Y], size[Z]);
  
  // (1) Zero-padding in Z direction
  /// @todo: only if necessary
  gpu_zero(buffer1, size[X]*size[Y]*paddedSize[Z]);
  timer_start("copy_pad_1");
  gpu_copy_pad(input, buffer1, size[X], size[Y], size[Z], size[X], size[Y], paddedSize[Z]);
  gpu_sync();
  timer_stop("copy_pad_1");

  //print("buffer1", buffer1, size[X], size[Y], paddedSize[Z]);
    
  // (2) Out-of-place R2C FFT Z
  timer_start("FFT_R2C");
  gpu_safefft( cufftExecR2C(plan->fwPlanZ, (cufftReal*)buffer1,  (cufftComplex*)buffer2) );
  gpu_sync();
  timer_stop("FFT_R2C");

  //printc("buffer2", buffer2, size[X], size[Y], complexZ, 2);
  
  // (3) transpose Y-Z
  timer_start("transposeYZ");
  gpu_transposeYZ_complex(buffer2, buffer2t, size[X], size[Y], complexZ*2);
  gpu_sync();
  timer_stop("transposeYZ");


  //printc("buffer2t", buffer2t, size[X], complexZ, size[Y], 2);
    
  // (4) Zero-padding in Z'
  gpu_zero(buffer3, size[X]*complexZ*paddedSize[Y]*2);
  timer_start("copy_pad_2");
  gpu_copy_pad(buffer2t, buffer3, size[X], complexZ, size[Y]*2, size[X], complexZ, paddedSize[Y]*2);
  gpu_sync();
  timer_stop("copy_pad_2");

  //printc("buffer3", buffer3, size[X], complexZ, paddedSize[Y], 2);
  
  // (5) In-place C2C FFT Y
  timer_start("FFT_Y");
  gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)buffer3,  (cufftComplex*)buffer3, CUFFT_FORWARD) );
  gpu_sync();
  timer_stop("FFT_Y");

  //printc("buffer3", buffer3, size[X], complexZ, paddedSize[Y], 2);
  
  if(paddedSize[X] == 1){
    ///@todo: copy not neccesary, use output instead of buffer3 in previous step
    memcpy_on_gpu(buffer3, output, size[X] * complexZ * paddedSize[Y] * 2); //buffer3size
  }else{
    // (6) Transpose X-Z
    timer_start("transposeXZ");
    gpu_transposeXZ_complex(buffer3, buffer3t, size[X], complexZ, paddedSize[Y]*2);
    gpu_sync();
    timer_stop("transposeXZ");

    //printc("buffer3t", buffer3t, paddedSize[Y], complexZ, size[X], 2);

    // (7) Zero-padding in Z''
    gpu_zero(output, paddedSize[Y]*complexZ*paddedSize[X]*2);
    timer_start("copy_pad_3");
    gpu_copy_pad(buffer3t, output, paddedSize[Y], complexZ, size[X]*2, paddedSize[Y], complexZ, paddedSize[X]*2);
    gpu_sync();
    timer_stop("copy_pad_3");

    //printc("output", output, paddedSize[Y], complexZ, paddedSize[X], 2);

    // (8) In-place C2C FFT X
    timer_start("FFT_X");
    gpu_safefft( cufftExecC2C(plan->planX, (cufftComplex*)output,  (cufftComplex*)output, CUFFT_FORWARD) );
    gpu_sync();
    timer_stop("FFT_X");

    //printc("output", output, paddedSize[Y], complexZ, paddedSize[X], 2);
  }
>>>>>>> arne
}




<<<<<<< HEAD




void yz_transpose_in_place_fw(float *data, int *size, int *pSSize){

  int offset = pSSize[X]*pSSize[Y]*pSSize[Z]/2;    //start of second half
  int pSSize_YZ = pSSize[Y]*pSSize[Z];

  if (size[X]!=pSSize[X]){
    for (int i=0; i<size[X]; i++){       // transpose each plane out of place: can be parallellized
      int ind1 = offset + i*size[Y]*pSSize[Z];
      int ind2 = i*pSSize_YZ;
      gpu_transpose_complex(data + ind1, data + ind2, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y]);
    }
    gpu_zero(data + offset, offset);     // possible to delete values in gpu_transpose_complex
  }
  else{     //padding in the y-direction
    for (int i=0; i<size[X]-1; i++){       // transpose all but the last plane out of place: can only partly be parallellized
      int ind1 = offset + i*size[Y]*pSSize[Z];
      int ind2 = i*pSSize_YZ;
      gpu_transpose_complex(data + ind1, data + ind2, size[Y], pSSize[Z], 0, pSSize[Y]-size[Y]);
      gpu_zero(data + offset + i*size[Y]*pSSize[Z], size[Y]*pSSize[Z]);     // deletable
    }
    gpu_transpose_complex_in_plane_fw(data + (size[X]-1)*pSSize_YZ, size[Y], pSSize[Z]);
  }
  

  return;
}


void yz_transpose_in_place_inv(float *data, int *size, int *pSSize){

  int offset = pSSize[X]*pSSize[Y]*pSSize[Z]/2;    //start of second half
  int pSSize_YZ = pSSize[Y]*pSSize[Z];

  if (size[X]!=pSSize[X])
      // transpose each plane out of place: can be parallellized
    for (int i=0; i<size[X]; i++){
      int ind1 = i*pSSize_YZ;
      int ind2 = offset + i*size[Y]*pSSize[Z];
      gpu_transpose_complex(data + ind1, data + ind2, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0);
    }
  else{
      // last plane needs to transposed in plane
    gpu_transpose_complex_in_plane_inv(data + (size[X]-1)*pSSize_YZ, pSSize[Z]/2, 2*size[Y]);
      // transpose all but the last plane out of place: can only partly be parallellized
    for (int i=0; i<size[X]-1; i++){
      int ind1 = i*pSSize_YZ;
      int ind2 = offset + i*size[Y]*pSSize[Z];
      gpu_transpose_complex(data + ind1, data + ind2, pSSize[Z]/2, 2*size[Y], pSSize[Y]-size[Y], 0);

    }
  }
  
  return;
=======
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, float* input, float* output){
  
  int complexZ = (plan->paddedSize[Z] + 2)/2;
  int* size = plan->size;
  int* paddedSize = plan->paddedSize;
//   int* paddedComplexSize = plan->paddedComplexSize;
  float* buffer1 = plan->buffer1;
  float* buffer2 = plan->buffer2;
  float* buffer2t = plan->buffer2t;
  float* buffer3 = plan->buffer3;
  float* buffer3t = plan->buffer3t;

  //printc("input", input, paddedSize[Y], complexZ, paddedSize[X], 2);
  if(paddedSize[X] == 1){
    ///@todo: copy not neccesary, use output instead of buffer3 in previous step
    memcpy_on_gpu(input, buffer3, size[X] * complexZ * paddedSize[Y] * 2); //buffer3size
  }else{
    // (8) In-place C2C FFT X
    timer_start("-FFT_X");
    gpu_safefft( cufftExecC2C(plan->planX, (cufftComplex*)input,  (cufftComplex*)input, CUFFT_INVERSE) );
    gpu_sync();
    timer_stop("-FFT_X");

    //printc("input", input, paddedSize[Y], complexZ, paddedSize[X], 2);

    // (7) Zero-padding in Z''
    gpu_zero(buffer3t, paddedSize[Y]*complexZ*size[X]*2);
    timer_start("-copy_pad_3");
    gpu_copy_unpad(input, buffer3t,   paddedSize[Y], complexZ, paddedSize[X]*2,   paddedSize[Y], complexZ, size[X]*2);
    gpu_sync();
    timer_stop("-copy_pad_3");

    //printc("buffer3t", buffer3t, paddedSize[Y], complexZ, size[X], 2);

    // (6) Transpose X-Z
    timer_start("-transposeXZ");
    gpu_transposeXZ_complex(buffer3t, buffer3, paddedSize[Y], complexZ, size[X]*2);
    gpu_sync();
    timer_stop("-transposeXZ");
  }
  //printc("buffer3", buffer3, size[X], complexZ, paddedSize[Y], 2);
    
  // (5) In-place C2C FFT Y
  timer_start("-FFT_Y");
  gpu_safefft( cufftExecC2C(plan->planY, (cufftComplex*)buffer3,  (cufftComplex*)buffer3, CUFFT_INVERSE) );
  gpu_sync();
  timer_stop("-FFT_Y");

  //printc("buffer3", buffer3, size[X], complexZ, paddedSize[Y], 2);
    
  // (4) Zero-padding in Z'
  gpu_zero(buffer2t, size[X]*complexZ*size[Y]*2);
  timer_start("-copy_pad_2");
  gpu_copy_unpad(buffer3, buffer2t,   size[X], complexZ, paddedSize[Y]*2,    size[X], complexZ, size[Y]*2);
  gpu_sync();
  timer_stop("-copy_pad_2");

  //printc("buffer2t", buffer2t, size[X], complexZ, size[Y], 2);
   
  // (3) transpose Y-Z
  timer_start("-transposeYZ");
  gpu_transposeYZ_complex(buffer2t, buffer2,   size[X], complexZ, size[Y]*2);
  gpu_sync();
  timer_stop("-transposeYZ");

  //printc("buffer2", buffer2, size[X], size[Y], complexZ, 2);

  // (2) Out-of-place R2C FFT Z
  timer_start("-FFT_C2R");
  gpu_safefft( cufftExecC2R(plan->invPlanZ, (cufftComplex*)buffer2,  (cufftReal*)buffer1) );
  gpu_sync();
  timer_stop("-FFT_C2R");

  //print("buffer1", buffer1, size[X], size[Y], paddedSize[Z]);
    
  // (1) Zero-padding in Z direction
  gpu_zero(output, size[X]*size[Y]*size[Z]);  // not neccesary for unpad?
  timer_start("-copy_pad_1");
  gpu_copy_unpad(buffer1, output,   size[X], size[Y], paddedSize[Z],   size[X], size[Y], size[Z]);
  gpu_sync();
  timer_stop("-copy_pad_1");

  //print("output", output, size[X], size[Y], size[Z]);

}


int gpuFFT3dPlan_normalization(gpuFFT3dPlan* plan){
  return plan->paddedSize[X] * plan->paddedSize[Y] * plan->paddedSize[Z];
>>>>>>> arne
}


#ifdef __cplusplus
}
#endif