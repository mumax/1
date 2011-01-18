#include "gpu_init.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Not much setup is needed here, only the number of threads per block is optionally set
void gpu_init(int threads,  ///< number of threads per block, 0 means autoset
              int options   ///< currently not used
              ){
  gpu_setmaxthreads(threads);
}

void gpu_set_device(int devid){
  gpu_safe(cudaSetDevice(devid));
}

#ifdef __cplusplus
}
#endif
