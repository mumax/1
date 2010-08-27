#include "gpu_safe.h"
#include <stdio.h>
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif


// void gpu_safe(int status){
//   if(status != cudaSuccess){
//     fprintf(stderr, "received CUDA error: %s\n", cudaGetErrorString((cudaError_t)status));
//     abort();
//   }
// }


char* cufftGetErrorString(cufftResult s){
  switch(s){
    default: return "Unknown error";
    case CUFFT_SUCCESS: return "Any CUFFT operation is successful.";
    case CUFFT_INVALID_PLAN: return "CUFFT is passed an invalid plan handle.";
    case CUFFT_ALLOC_FAILED: return "CUFFT failed to allocate GPU memory.";
    case CUFFT_INVALID_TYPE: return "The user requests an unsupported type.";
    case CUFFT_INVALID_VALUE: return "The user specifies a bad memory pointer.";
    case CUFFT_INTERNAL_ERROR: return "Used for all internal driver errors.";
    case CUFFT_EXEC_FAILED: return "CUFFT failed to execute an FFT on the GPU.";
    case CUFFT_SETUP_FAILED: return "The CUFFT library failed to initialize.";
//     case CUFFT_SHUTDOWN_FAILED: return "The CUFFT library failed to shut down.";
    case CUFFT_INVALID_SIZE: return "The user specifies an unsupported FFT size.";
  }
}

#ifdef __cplusplus
}
#endif
