#include "gpu_init.h"
#include "gpu_safe.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Doesn't do much for the moment, but here for uniformity with CPU, where init() _is_ neccesary (to initialize FFTW)
void gpu_init(){
  
}

void gpu_set_device(int devid){
  gpu_safe(cudaSetDevice(devid));
}

#ifdef __cplusplus
}
#endif
