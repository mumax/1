#include "gpu_bigfft.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_bigfft(bigfft* target, int size, cufftType type, int batch){
  assert(size <= MAX_FFTSIZE);
  
  fprintf(stderr, "init_bigfft(%p, %d, %d)\n", target, size, batch);
  target->maxBatch = MAX_FFTSIZE / size;  // the maximum number of batches we can run in one plan
  fprintf(stderr, "maxBatch=%d\n", target->maxBatch);
  
  target->nPlan1 = batch / target->maxBatch;
  if (target->nPlan1 > 0){
    fprintf(stderr, "plan1 = cufftPlan1d(%d, %d); nPlan1 = %d\n", size, target->maxBatch, target->nPlan1);
    gpu_safefft( cufftPlan1d(&(target->plan1), size, type, target->maxBatch) );
    gpu_safefft( cufftSetCompatibilityMode((target->plan1), CUFFT_COMPATIBILITY_NATIVE) );
  }

  int batch2 = batch % target->maxBatch;
  if(batch2 > 0){
    target->nPlan2 = 1;
  }
  else{
    target->nPlan2 = 0;
  }
  
  if (target->nPlan2 > 0){
    fprintf(stderr, "plan2 = cufftPlan1d(%d, %d); nPlan2 = %d\n", size, batch2, target->nPlan2);
    gpu_safefft( cufftPlan1d(&(target->plan2), size, type, batch2) );
    gpu_safefft( cufftSetCompatibilityMode((target->plan2), CUFFT_COMPATIBILITY_NATIVE) );
  }
}

void bigfft_execR2C(bigfft* plan, cufftReal* input, cufftComplex* output){
  
  int in_offset = 0, out_offset = 0; // offset indices for taking sub-arrays of the input/output data
  
  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecR2C(plan->plan1,&(input[in_offset]), &(output[out_offset])) );
    in_offset += plan->maxBatch * (plan->size);
    out_offset += plan->maxBatch * (plan->size + 1);
  }
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecR2C(plan->plan2, &(input[in_offset]), &(output[out_offset])) );
  }
}

void bigfft_execC2R(bigfft* plan, cufftComplex* input, cufftReal* output){

  int in_offset = 0, out_offset = 0;

  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecC2R(plan->plan1, &(input[in_offset]), &(output[out_offset])) );
    in_offset += plan->maxBatch * (plan->size + 1);
    out_offset += plan->maxBatch * (plan->size);
  }
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecC2R(plan->plan2, &(input[in_offset]), &(output[out_offset])) );
  }
}


void bigfft_execC2C(bigfft* plan, cufftComplex* input, cufftComplex* output, int direction){

  int in_offset = 0, out_offset = 0;

  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecC2C(plan->plan1, &(input[in_offset]), &(output[out_offset]), direction) );
    in_offset += plan->maxBatch * (plan->size);
    out_offset += plan->maxBatch * (plan->size);
  }
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecC2C(plan->plan2, &(input[in_offset]), &(output[out_offset]), direction) );
  }
}


#ifdef __cplusplus
}
#endif
