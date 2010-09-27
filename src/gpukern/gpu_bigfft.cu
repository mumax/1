#include "gpu_bigfft.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_bigfftR2C(bigfft* target, int size, int batch){
  gpu_safefft( cufftPlan1d(&(target->plan), size, CUFFT_R2C, batch) );
  gpu_safefft( cufftSetCompatibilityMode((target->plan), CUFFT_COMPATIBILITY_NATIVE) );
}

void init_bigfftC2R(bigfft* target, int size, int batch){
  gpu_safefft( cufftPlan1d(&(target->plan), size, CUFFT_C2R, batch) );
  gpu_safefft( cufftSetCompatibilityMode((target->plan), CUFFT_COMPATIBILITY_NATIVE) );
}

void init_bigfftC2C(bigfft* target, int size, int batch){
  gpu_safefft( cufftPlan1d(&(target->plan), size, CUFFT_C2C, batch) );
  gpu_safefft( cufftSetCompatibilityMode((target->plan), CUFFT_COMPATIBILITY_NATIVE) );
}

void bigfft_execR2C(bigfft* plan, float* input, float* output){
  gpu_safefft( cufftExecR2C(plan->plan, (cufftReal*)input, (cufftComplex*)output) );
}

void bigfft_execC2R(bigfft* plan, float* input, float* output){
  gpu_safefft( cufftExecC2R(plan->plan, (cufftComplex*)input, (cufftReal*)output) );
}

void bigfft_execC2C(bigfft* plan, float* input, float* output, int direction){
  gpu_safefft( cufftExecC2C(plan->plan, (cufftComplex*)input, (cufftComplex*)output, direction) );
}


#ifdef __cplusplus
}
#endif
