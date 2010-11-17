#include "gpu_fft0.h"
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


/**
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @TODO interleave with streams: fft_z(MX), transpose(MX) async, FFT_z(MY), transpose(MY) async, ... threadsync, FFT_Y(MX)...
 * @TODO CUFFT crashes on more than one million cells, seems to be exactly the limit of 8 million elements (after zero-padding + 2 extra rows)
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size, int* paddedSize){
  
  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  gpuFFT3dPlan* plan = (gpuFFT3dPlan*)malloc(sizeof(gpuFFT3dPlan));
  
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

  plan->paddedComplexN = plan->paddedSize[X] * plan->paddedSize[Y] * complexZ;
  

  if (paddedSize[X] > 1){
    gpu_safefft(cufftPlan3d(&(plan->fwPlan), paddedSize[X], paddedSize[Y], paddedSize[Z], CUFFT_R2C))
    gpu_safefft(cufftPlan3d(&(plan->invPlan), paddedSize[X], paddedSize[Y], paddedSize[Z], CUFFT_C2R))
  }else{
    gpu_safefft(cufftPlan2d(&(plan->fwPlan),  paddedSize[Y], paddedSize[Z], CUFFT_R2C))
    gpu_safefft(cufftPlan2d(&(plan->invPlan), paddedSize[Y], paddedSize[Z], CUFFT_C2R))
  }

  plan->buffer1 = new_gpu_array(paddedSize[X] * paddedSize[Y] * complexZ);

  return plan;
}


gpuFFT3dPlan* new_gpuFFT3dPlan(int* size){
  return new_gpuFFT3dPlan_padded(size, size); // when size == paddedsize, there is no padding
}

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, float* input, float* output){
    //gpu_zero(plan->buffer1, plan->paddedComplexN);
    //gpu_copy_pad(input, plan->buffer1, plan->size[X], plan->size[Y], plan->size[Z], plan->paddedSize[X], plan->paddedSize[Y], plan->paddedSize[Z]);

    //memcpy_on_gpu(input, plan->buffer1, plan->paddedSize[X]*plan->paddedSize[Y]*plan->paddedSize[Z]);
    //gpu_safefft( cufftExecR2C(plan->fwPlan, (cufftReal*)plan->buffer1, (cufftComplex*)output) );

//     gpu_safefft( cufftExecR2C(plan->fwPlan, (cufftReal*)input, (cufftComplex*)output) );

  memcpy_on_gpu(input, output, plan->paddedSize[X]*plan->paddedSize[Y]*plan->paddedSize[Z]);

}



void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan, float* input, float* output){
  //gpu_safefft( cufftExecC2R(plan->invPlan, (cufftComplex*)input, (cufftReal*)output) );

  //gpu_copy_unpad(plan->buffer1, output, plan->paddedSize[X], plan->paddedSize[Y], (plan->paddedSize[Z]/2+2), plan->size[X], plan->size[Y], plan->size[Z]);
  //memcpy_on_gpu(plan->buffer1, output, plan->paddedSize[X]*plan->paddedSize[Y]*plan->paddedSize[Z]);

  memcpy_on_gpu(input, output, plan->paddedSize[X]*plan->paddedSize[Y]*plan->paddedSize[Z]);
}


int gpuFFT3dPlan_normalization(gpuFFT3dPlan* plan){
  return plan->paddedSize[X] * plan->paddedSize[Y] * plan->paddedSize[Z];
}


#ifdef __cplusplus
}
#endif