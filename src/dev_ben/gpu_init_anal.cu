#include "gpu_init_anal.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif


gpuanalfw* new_gpuanalfw(param* p){
  
  check_param(p);
  gpuanalfw* anal_fw = (gpuanalfw*) malloc(sizeof(gpuanalfw));
  anal_fw->params = p;
  
  return anal_fw;
}

gpuanalpc* new_gpuanalpc(param* p){
  
  check_param(p);
  int* size4D = tensor_size4D(p->size);
  
  gpuanalpc* anal_pc = (gpuanalpc*) malloc(sizeof(gpuanalpc));
  anal_pc->params = p;
  anal_pc->m2 = new_gputensor(4, size4D);
  anal_pc->h2 = new_gputensor(4, size4D);
  return anal_pc;
}


#ifdef __cplusplus
}
#endif