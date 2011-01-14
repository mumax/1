#include "cpu_fft1.h"
#include "cpu_mem.h"
#include "cpu_zeropad.h"
#include "fftw3.h"
#include "../macros.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int fftw_strategy;

/**
 * @internal 
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 */

void init_cpu_plans(cpuFFT3dPlan *plan){

  int FFTdimx = plan->paddedSize[X];
  int FFTdimy = plan->paddedSize[Y];
  int FFTdimz1 = plan->paddedSize[Z];
  int FFTdimz2 = plan->paddedSize[Z]+2;
  int FFTdimyz = FFTdimy*FFTdimz2;
  
  int paddedN = plan->paddedSize[X] * plan->paddedSize[Y] * (plan->paddedSize[Z]+2);
  float* array = new_cpu_array(paddedN);
  fftwf_complex *carray = (fftwf_complex *) array;
  
  plans->FFT_FW_dim3 =
    fftwf_plan_many_dft_r2c (1, &FFTdimz1, dimy, array, NULL, 1, FFTdimz2,
      carray, NULL, 1, FFTdimz2/2, fftw_strategy);

  plans->FFT_FW_dim2 =
    fftwf_plan_many_dft (1, &FFTdimy, FFTdimz2/2, carray, NULL, FFTdimz2/2, 1,
      carray, NULL, FFTdimz2/2, 1, 1, fftw_strategy);

  if (FFTdimx>1){
    plans->FFT_FW_dim1 =
      fftwf_plan_many_dft (1, &FFTdimx, FFTdimyz/2, carray, NULL, FFTdimyz/2, 1,
        carray, NULL, FFTdimyz/2, 1, 1, fftw_strategy);
    plans->FFT_BW_dim1 =
      fftwf_plan_many_dft (1, &FFTdimx, FFTdimyz/2, carray, NULL, FFTdimyz/2, 1,
        carray, NULL, FFTdimyz/2, 1, -1, fftw_strategy);
  }

  plans->FFT_BW_dim2 =
    fftwf_plan_many_dft (1, &FFTdimy, FFTdimz2/2, carray, NULL, FFTdimz2/2, 1,
      carray, NULL, FFTdimz2/2, 1, -1, fftw_strategy);

  plans->FFT_BW_dim3 =
    fftwf_plan_many_dft_c2r (1, &FFTdimz1, dimy, carray, NULL, 1, FFTdimz2/2,
      array, NULL, 1, FFTdimz2, fftw_strategy);
      
  free (array);

  return;
}

cpuFFT3dPlan* new_cpuFFT3dPlan_padded(int* size, int* paddedSize){
  
  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  cpuFFT3dPlan* plan = (cpuFFT3dPlan*)malloc(sizeof(cpuFFT3dPlan));
  
  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  
  plan->size[X] = N0; 
  plan->size[Y] = N1; 
  plan->size[Z] = N2;
  
  plan->paddedSize[X] = paddedSize[X];
  plan->paddedSize[Y] = paddedSize[Y];
  plan->paddedSize[Z] = paddedSize[Z];

  init_cpu_plans(plan);
  
  return plan;
}

void delete_cpuFFT3dPlan(cpuFFT3dPlan* plan){

  if (plan->size[X]>1){
    fftwf_destroy_plan (plan->FFT_FW_dim1);
    fftwf_destroy_plan (plan->FFT_BW_dim1);
  }
  fftwf_destroy_plan (plan->FFT_FW_dim2);
  fftwf_destroy_plan (plan->FFT_BW_dim2);
  fftwf_destroy_plan (plan->FFT_FW_dim3);
  fftwf_destroy_plan (plan->FFT_BW_dim3);

  free (plan->size);
  free (plan->paddedSize);
  free (plan);
  
  return;
}

void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan, float* input, float* output){

  cpu_copy_pad(input, output, plan->size[X], plan->size[Y], plan->size[Z], plan->paddedSize[X], plan->paddedSize[Y], (plan->paddedSize[Z]+2));
  
  int FFTdimy = plan->paddedSize[Y];
  int FFTdimz = plan->paddedSize[Z]+2;
  int FFTdimyz = FFTdimy*FFTdimz;
  fftwf_complex *cFFTout = (fftwf_complex *) output;

  for (int cnt=0; cnt<plan->size[X]; cnt++)
    fftwf_execute_dft_r2c(plans->FFT_FW_dim3, &output[cnt*FFTdimyz], &cFFTout[cnt*FFTdimyz/2]);

  for (int cnt=0; cnt<plan->size[X]; cnt++)
    fftwf_execute_dft(plans->FFT_FW_dim2, &cFFTout[cnt*FFTdimyz/2], &cFFTout[cnt*FFTdimyz/2]);

  if (plan->size[X]>1)
    fftwf_execute_dft(plans->FFT_FW_dim1, cFFTout, cFFTout);

  return;

}

void cpuFFT3dPlan_inverse(cpuFFT3dPlan* plan, float* input, float* output){

  int FFTdimy = plan->paddedSize[Y];
  int FFTdimz = plan->paddedSize[Z]+2;
  int FFTdimyz = FFTdimy*FFTdimz;
  fftwf_complex *cFFTin = (fftwf_complex *) input;
  
  
  if (plan->size[X]>1)
    fftwf_execute_dft(plans->FFT_BW_dim1, cFFTin, cFFTin);

  for (int cnt=0; cnt<plan->size[X]; cnt++)
    fftwf_execute_dft(plans->FFT_BW_dim2, &cFFTin[cnt*FFTdimyz/2], &cFFTin[cnt*FFTdimyz/2]);

  for (int cnt=0; cnt<plan->size[X]; cnt++)
    fftwf_execute_dft_c2r(plans->FFT_BW_dim3, &cFFTin[cnt*FFTdimyz/2], &input[cnt*FFTdimyz]);

  cpu_copy_unpad(input, output, plan->paddedSize[X], plan->paddedSize[Y], (plan->paddedSize[Z]+2), plan->size[X], plan->size[Y], plan->size[Z]);
}


#ifdef __cplusplus
}
#endif
