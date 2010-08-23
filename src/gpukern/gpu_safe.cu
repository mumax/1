#include "gpu_safe.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


// void gpu_safe(int status){
//   if(status != cudaSuccess){
//     fprintf(stderr, "received CUDA error: %s\n", cudaGetErrorString((cudaError_t)status));
//     abort();
//   }
// }

#ifdef __cplusplus
}
#endif
