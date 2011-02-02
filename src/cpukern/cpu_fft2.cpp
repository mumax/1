#include "cpu_fft2.h"
#include "cpu_mem.h"
#include "cpu_zeropad.h"
#include "../macros.h"
#include <stdlib.h>



#ifdef __cplusplus
extern "C" {
#endif

extern int fftw_strategy;

/**
 * @internal 
 * Creates a new FFT plan for transforming the magnetization. 
 * zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.  Multithreaded!
 */
void init_cpu_plans(cpuFFT3dPlan *plan){

  int FFTdimx = plan->paddedSize[0];
  int FFTdimy = plan->paddedSize[1];
  int FFTdimz1 = plan->paddedSize[2];
  int FFTdimz2 = plan->paddedSize[2]+2;
  int FFTdimyz = FFTdimy*FFTdimz2;

  int paddedN = plan->paddedSize[0] * plan->paddedSize[1] * (plan->paddedSize[2]+2);
  float* array = new_cpu_array(paddedN);
  fftwf_complex *carray = (fftwf_complex *) array;

  int Nthreads = plan->T_data->N_threads;
  
  int N_temp1[Nthreads];

  int n = plan->size[0]/Nthreads;
  for (int cnt=0; cnt<Nthreads; cnt++)
    N_temp1[cnt] = n;
  for (int cnt=0; cnt<plan->size[0] - n*Nthreads; cnt++)
    N_temp1[cnt]++;

  plan->N_FFT_dimx = (int *) calloc(Nthreads+1, sizeof(int));
  plan->N_FFT_dimx[0] = 0;
  for (int cnt=0; cnt<Nthreads; cnt++)
    plan->N_FFT_dimx[cnt+1] = plan->N_FFT_dimx[cnt] + N_temp1[cnt];

  n = FFTdimyz/2/Nthreads;
  for (int cnt=0; cnt<Nthreads; cnt++)
    N_temp1[cnt] = n;
  for (int cnt=0; cnt<FFTdimyz/2 - n*Nthreads; cnt++)
    N_temp1[cnt]++;

  plan->N_FFT_dimyz = (int *) calloc(Nthreads+1, sizeof(int));
  plan->N_FFT_dimyz[0] = 0;
  for (int cnt=0; cnt<Nthreads; cnt++)
    plan->N_FFT_dimyz[cnt+1] = plan->N_FFT_dimyz[cnt] + N_temp1[cnt];


  FILE *fftwisdom = fopen("fftw_wisdom","r");
  if ( fftwisdom!=NULL ){
    fftwf_import_wisdom_from_file(fftwisdom);
    fclose(fftwisdom);
  }


  plan->FFT_FW_dim3 =
    fftwf_plan_many_dft_r2c (1, &FFTdimz1, plan->size[1], array, NULL, 1, FFTdimz2,
      carray, NULL, 1, FFTdimz2/2, fftw_strategy);

  plan->FFT_FW_dim2 =
    fftwf_plan_many_dft (1, &FFTdimy, FFTdimz2/2, carray, NULL, FFTdimz2/2, 1,
      carray, NULL, FFTdimz2/2, 1, 1, fftw_strategy);
      
  if (FFTdimx>1){
    plan->FFT_FW_dim1 = (fftwf_plan *) calloc(Nthreads, sizeof(fftwf_plan));
    for (int cnt=0; cnt<Nthreads; cnt++)
      plan->FFT_FW_dim1[cnt] =
        fftwf_plan_many_dft (1, &FFTdimx, N_temp1[cnt],
          &carray[plan->N_FFT_dimyz[cnt]], NULL, FFTdimyz/2, 1,
          &carray[plan->N_FFT_dimyz[cnt]], NULL, FFTdimyz/2, 1, 1, fftw_strategy);

    plan->FFT_BW_dim1 = (fftwf_plan *) calloc(Nthreads, sizeof(fftwf_plan));
    for (int cnt=0; cnt<Nthreads; cnt++)
      plan->FFT_BW_dim1[cnt] =
        fftwf_plan_many_dft (1, &FFTdimx, N_temp1[cnt],
          &carray[plan->N_FFT_dimyz[cnt]], NULL, FFTdimyz/2, 1,
          &carray[plan->N_FFT_dimyz[cnt]], NULL, FFTdimyz/2, 1, -1, fftw_strategy);
  }
  
  plan->FFT_BW_dim2 =
    fftwf_plan_many_dft (1, &FFTdimy, FFTdimz2/2, carray, NULL, FFTdimz2/2, 1,
      carray, NULL, FFTdimz2/2, 1, -1, fftw_strategy);

  plan->FFT_BW_dim3 =
    fftwf_plan_many_dft_c2r (1, &FFTdimz1, plan->size[1], carray, NULL, 1, FFTdimz2/2,
      array, NULL, 1, FFTdimz2, fftw_strategy);

  fftwisdom = fopen("fftw_wisdom","w");
  fftwf_export_wisdom_to_file(fftwisdom);
  fclose(fftwisdom);

  return;
}

cpuFFT3dPlan *new_cpuFFT3dPlan(int* size, int* paddedSize){

  assert(paddedSize[0] > 0);
  assert(paddedSize[1] > 1);
  assert(paddedSize[2] > 1);

  cpuFFT3dPlan *plan = (cpuFFT3dPlan*) malloc(sizeof(cpuFFT3dPlan));

  plan->size = (int*)calloc(3, sizeof(int));
  plan->paddedSize = (int*)calloc(3, sizeof(int));
  
  plan->size[0] = size[0]; 
  plan->size[1] = size[1]; 
  plan->size[2] = size[2];
  
  plan->paddedSize[0] = paddedSize[0];
  plan->paddedSize[1] = paddedSize[1];
  plan->paddedSize[2] = paddedSize[2];

  plan->T_data = T_data;
  
  init_cpu_plans(plan);
  
  return(plan);
}

void delete_cpuFFT3dPlan (cpuFFT3dPlan *plan){
  
  if (plan->size[0]>1)
    for (int cnt=0; cnt<plan->T_data->N_threads; cnt++){
      fftwf_destroy_plan (plan->FFT_FW_dim1[cnt]);
      fftwf_destroy_plan (plan->FFT_BW_dim1[cnt]);
    }
  fftwf_destroy_plan (plan->FFT_FW_dim2);
  fftwf_destroy_plan (plan->FFT_BW_dim2);
  fftwf_destroy_plan (plan->FFT_FW_dim3);
  fftwf_destroy_plan (plan->FFT_BW_dim3);

  free (plan->N_FFT_dimx);
  free (plan->N_FFT_dimyz);
  free (plan->size);
  free (plan->paddedSize);
  free (plan);
  
  return;
}


typedef struct{
  
  cpuFFT3dPlan *plan;
  float *array;

} cpuFFT3dPlan_arg;

void cpuFFT3dPlan_forward_1(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  float *array = arg->array;
  
  int FFTdimy = plan->paddedSize[1];
  int FFTdimz = plan->paddedSize[2]+2;
  int FFTdimyz = FFTdimy*FFTdimz;
  fftwf_complex *carray = (fftwf_complex *) array;

  for (int cnt=plan->N_FFT_dimx[id]; cnt<plan->N_FFT_dimx[id+1]; cnt++)
    fftwf_execute_dft_r2c(plan->FFT_FW_dim3, &array[cnt*FFTdimyz], &carray[cnt*FFTdimyz/2]);

  return;
  }

void cpuFFT3dPlan_forward_2(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  fftwf_complex *carray = (fftwf_complex *) arg->array;

  int FFTdimy = plan->paddedSize[1];
  int FFTdimz = plan->paddedSize[2]+2;
  int FFTdimyz = FFTdimy*FFTdimz;

  for (int cnt=plan->N_FFT_dimx[id]; cnt<plan->N_FFT_dimx[id+1]; cnt++)
    fftwf_execute_dft(plan->FFT_FW_dim2, &carray[cnt*FFTdimyz/2], &carray[cnt*FFTdimyz/2]);

  return;
  }

void cpuFFT3dPlan_forward_3(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  fftwf_complex *carray = (fftwf_complex *) arg->array;

  fftwf_execute_dft(plan->FFT_FW_dim1[id],
    &carray[plan->N_FFT_dimyz[id]], &carray[plan->N_FFT_dimyz[id]]);

  return;
  }

void cpuFFT3dPlan_forward(cpuFFT3dPlan* plan, float* input, float* output){

  cpu_zero(output, plan->paddedSize[0]* plan->paddedSize[1]* (plan->paddedSize[2]+2));
  cpu_copy_pad(input, output, plan->size[0], plan->size[1], plan->size[2], plan->paddedSize[0], plan->paddedSize[1], (plan->paddedSize[2]+2));

  cpuFFT3dPlan_arg arg;
  arg.plan = plan;
  arg.array = output;
  
  func_arg = (void *) (&arg);
  
  thread_Wrapper(cpuFFT3dPlan_forward_1);
  thread_Wrapper(cpuFFT3dPlan_forward_2);
  if (plan->size[0]>1)
    thread_Wrapper(cpuFFT3dPlan_forward_3);

  return;
}


void cpuFFT3dPlan_inverse_1(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  fftwf_complex *carray = (fftwf_complex *) arg->array;

  fftwf_execute_dft(plan->FFT_BW_dim1[id],
    &carray[plan->N_FFT_dimyz[id]], &carray[plan->N_FFT_dimyz[id]]);

  return;
  }

void cpuFFT3dPlan_inverse_2(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  fftwf_complex *carray = (fftwf_complex *) arg->array;

  int FFTdimy = plan->paddedSize[1];
  int FFTdimz = plan->paddedSize[2]+2;
  int FFTdimyz = FFTdimy*FFTdimz;

  for (int cnt=plan->N_FFT_dimx[id]; cnt<plan->N_FFT_dimx[id+1]; cnt++)
    fftwf_execute_dft(plan->FFT_BW_dim2, &carray[cnt*FFTdimyz/2], &carray[cnt*FFTdimyz/2]);

  return;
  }

void cpuFFT3dPlan_inverse_3(int id){

  cpuFFT3dPlan_arg *arg = (cpuFFT3dPlan_arg *) func_arg;
  cpuFFT3dPlan *plan = arg->plan;
  float *array = arg->array;
  fftwf_complex *carray = (fftwf_complex *) array;

  int FFTdimy = plan->paddedSize[1];
  int FFTdimz = plan->paddedSize[2]+2;
  int FFTdimyz = FFTdimy*FFTdimz;

  for (int cnt=plan->N_FFT_dimx[id]; cnt<plan->N_FFT_dimx[id+1]; cnt++)
    fftwf_execute_dft_c2r(
      plan->FFT_BW_dim3, &carray[cnt*FFTdimyz/2], &array[cnt*FFTdimyz]);

  return;
  }

void cpuFFT3dPlan_inverse (cpuFFT3dPlan* plan, float* input, float* output){

  cpuFFT3dPlan_arg arg;
  arg.plan = plan;
  arg.array = input;
  
  func_arg = (void *) (&arg);

  if (plan->size[0]>1)
    thread_Wrapper(cpuFFT3dPlan_inverse_1);
  thread_Wrapper(cpuFFT3dPlan_inverse_2);
  thread_Wrapper(cpuFFT3dPlan_inverse_3);

  cpu_copy_unpad(input, output, plan->paddedSize[0], plan->paddedSize[1], (plan->paddedSize[2]+2), plan->size[0], plan->size[1], plan->size[2]);

  return;
}


#ifdef __cplusplus
}
#endif
