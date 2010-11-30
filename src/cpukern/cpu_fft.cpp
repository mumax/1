#include "cpu_fft.h"
#include "cpu_mem.h"
#include "cpu_zeropad.h"
#include "fftw3.h"
#include "../macros.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

///@todo cleanup

/**
 * @internal 
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 * @todo on compute capability < 2.0, the first step is done serially...
 * @todo rename kernelsize -> paddedsize
 */
cpuFFT3dPlan* new_cpuFFT3dPlan_padded(int* size, int* paddedSize, float* source, float* dest){
  
  int N0 = size[X];
  int N1 = size[Y];
  int N2 = size[Z];
  
  assert(paddedSize[X] > 0);
  assert(paddedSize[Y] > 1);
  assert(paddedSize[Z] > 1);
  
  cpuFFT3dPlan* plan = (cpuFFT3dPlan*)malloc(sizeof(cpuFFT3dPlan));
  
  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  
  plan->size[0] = N0; 
  plan->size[1] = N1; 
  plan->size[2] = N2;
  
  plan->paddedSize[X] = paddedSize[X];
  plan->paddedSize[Y] = paddedSize[Y];
  plan->paddedSize[Z] = paddedSize[Z];

  ///@todo Check for NULL return value: plan could not be created
  fftw_status status = fftw_import_wisdom_from_file(FILE *input_file);
  plan->fwPlan = fftwf_plan_dft_r2c_3d(paddedSize[X], paddedSize[Y], paddedSize[Z], source, (complex_t*)dest, FFTW_MEASURE | FFTW_USE_WISDOM); // replace by FFTW_PATIENT for super-duper performance
  plan->bwPlan = fftwf_plan_dft_c2r_3d(paddedSize[X], paddedSize[Y], paddedSize[Z], (complex_t*)source, dest, FFTW_MEASURE | FFTW_USE_WISDOM);
  fftw_export_wisdom_to_file(FILE *output_file);
  
  return plan;
}

void delete_cpuFFT3dPlan(cpuFFT3dPlan* plan){
  //TODO
}

cpuFFT3dPlan* new_cpuFFT3dPlan_outplace(int* datasize, int* paddedSize){
  int paddedN = paddedSize[X] * paddedSize[Y] * (paddedSize[Z]+2);
  float* in = new_cpu_array(paddedN);
  float* out = new_cpu_array(paddedN);
  cpuFFT3dPlan* plan = new_cpuFFT3dPlan_padded(datasize, paddedSize, in, out);
  free_cpu_array(in);
  free_cpu_array(out);
  return plan;
}


cpuFFT3dPlan* new_cpuFFT3dPlan_inplace(int* datasize, int* paddedSize){
  int paddedN = paddedSize[X] * paddedSize[Y] * (paddedSize[Z]+2);
  float* in = new_cpu_array(paddedN);
  cpuFFT3dPlan* plan = new_cpuFFT3dPlan_padded(datasize, paddedSize, in, in);
  free_cpu_array(in);
  return plan;
}


void cpuFFT3dPlan_forward(cpuFFT3dPlan* plan, float* input, float* output){
    cpu_zero(output, plan->paddedSize[X]* plan->paddedSize[Y]* (plan->paddedSize[Z]+2));
    cpu_copy_pad(input, output, plan->size[X], plan->size[Y], plan->size[Z], plan->paddedSize[X], plan->paddedSize[Y], (plan->paddedSize[Z]+2));
    fftwf_execute_dft_r2c((fftwf_plan) plan->fwPlan, output, (complex_t*)output);
}


void cpuFFT3dPlan_inverse(cpuFFT3dPlan* plan, float* input, float* output){
  fftwf_execute_dft_c2r((fftwf_plan) plan->bwPlan, (complex_t*)input, input);
  cpu_copy_unpad(input, output, plan->paddedSize[X], plan->paddedSize[Y], (plan->paddedSize[Z]+2), plan->size[X], plan->size[Y], plan->size[Z]);
}


#ifdef __cplusplus
}
#endif
