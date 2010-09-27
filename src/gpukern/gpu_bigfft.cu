#include "gpu_bigfft.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_bigfft(bigfft* target, int size, cufftType type, int batch){
  fprintf(stderr, "init_bigfft(%p, %d, %d)\n", target, size, batch);
  int maxBatch = MAX_FFTSIZE / size;  // the maximum number of batches we can run in one plan
  fprintf(stderr, "maxBatch=%d\n", maxBatch);
  
  target->nPlan1 = batch / maxBatch;
  if (target->nPlan1 > 0){
    fprintf(stderr, "plan1 = cufftPlan1d(%d, %d)\n", size, batch);
    gpu_safefft( cufftPlan1d(&(target->plan1), size, type, maxBatch) );
    gpu_safefft( cufftSetCompatibilityMode((target->plan1), CUFFT_COMPATIBILITY_NATIVE) );
  }

  int batch2 = batch % maxBatch;
  if(batch2 > 0){
    target->nPlan2 = 1;
  }
  else{
    target->nPlan2 = 0;
  }
  
  if (target->nPlan2 > 0){
    fprintf(stderr, "plan2 = cufftPlan1d(%d, %d)\n", size, batch2);
    gpu_safefft( cufftPlan1d(&(target->plan2), size, type, batch2) );
    gpu_safefft( cufftSetCompatibilityMode((target->plan2), CUFFT_COMPATIBILITY_NATIVE) );
  }
}


void bigfft_execR2C(bigfft* plan, cufftReal* input, cufftComplex* output){
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecR2C(plan->plan2, (cufftReal*)input, (cufftComplex*)output) );
  }
  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecR2C(plan->plan1, (cufftReal*)input, (cufftComplex*)output) );
  }
}

void bigfft_execC2R(bigfft* plan, cufftComplex* input, cufftReal* output){
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecC2R(plan->plan2, (cufftComplex*)input, (cufftReal*)output) );
  }
  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecC2R(plan->plan1, (cufftComplex*)input, (cufftReal*)output) );
  }
}


void bigfft_execC2C(bigfft* plan, cufftComplex* input, cufftComplex* output, int direction){
  for(int i=0; i<plan->nPlan2; i++){
    gpu_safefft( cufftExecC2C(plan->plan2, (cufftComplex*)input, (cufftComplex*)output, direction) );
  }
  for(int i=0; i<plan->nPlan1; i++){
    gpu_safefft( cufftExecC2C(plan->plan1, (cufftComplex*)input, (cufftComplex*)output, direction) );
  }
}


#ifdef __cplusplus
}
#endif
