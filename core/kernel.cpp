#include "kernel.h"

#ifdef __cplusplus
extern "C" {
#endif


tensor* new_kernel(param* params){

   tensor* kernel = NULL;
  
   int kernelType = params->kernelType;
   if(kernelType == KERNEL_MICROMAG3D){
      //kernel = gpu_micromag3d_kernel(params);
   }
   else if(kernelType == KERNEL_MICROMAG2D){
      //..
   }
   else{
      fprintf(stderr, "Unknown kernel type: %d\n", kernelType);
      abort();
   }

   return kernel;

}


#ifdef __cplusplus
}
#endif