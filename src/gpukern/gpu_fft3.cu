#include "gpu_fft3.h"
#include "../macros.h"
#include "gpu_transpose.h"
#include "gpu_safe.h"
#include "gpu_conf.h"
#include "gpu_mem.h"
#include "gpu_zeropad.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

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
gpuFFT3dPlanArne* new_gpuFFT3dPlanArne_padded(int* size, int* paddedSize){
  
  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  gpuFFT3dPlanArne* plan = (gpuFFT3dPlanArne*)malloc(sizeof(gpuFFT3dPlanArne));
  
  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
  
  plan->paddedSize[X] = paddedSize[X];
  plan->paddedSize[Y] = paddedSize[Y];
  plan->paddedSize[Z] = paddedSize[Z];
  plan->paddedN = plan->paddedSize[0] * plan->paddedSize[1] * plan->paddedSize[2];

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


gpuFFT3dPlanArne* new_gpuFFT3dPlanArne(int* size){
  return new_gpuFFT3dPlanArne_padded(size, size); // when size == paddedsize, there is no padding
}

void gpuFFT3dPlanArne_forward(gpuFFT3dPlanArne* plan, float* input, float* output){

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
}




void gpuFFT3dPlanArne_inverse(gpuFFT3dPlanArne* plan, float* input, float* output){
  
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


int gpuFFT3dPlanArne_normalization(gpuFFT3dPlanArne* plan){
  return plan->paddedSize[X] * plan->paddedSize[Y] * plan->paddedSize[Z];
}


#ifdef __cplusplus
}
#endif