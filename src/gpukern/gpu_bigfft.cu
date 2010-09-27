#include "gpu_bigfft.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_bigfft(bigfft* target, int size, cufftType type, int batch){
  gpu_safefft( cufftPlan1d(&(target->plan), size, type, batch) );
  gpu_safefft( cufftSetCompatibilityMode((target->plan), CUFFT_COMPATIBILITY_NATIVE) );
}


void bigfft_execR2C(bigfft* plan, cufftReal* input, cufftComplex* output){
  gpu_safefft( cufftExecR2C(plan->plan, (cufftReal*)input, (cufftComplex*)output) );
}

void bigfft_execC2R(bigfft* plan, cufftComplex* input, cufftReal* output){
  gpu_safefft( cufftExecC2R(plan->plan, (cufftComplex*)input, (cufftReal*)output) );
}

void bigfft_execC2C(bigfft* plan, cufftComplex* input, cufftComplex* output, int direction){
  gpu_safefft( cufftExecC2C(plan->plan, (cufftComplex*)input, (cufftComplex*)output, direction) );
}


#ifdef __cplusplus
}
#endif
