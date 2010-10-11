#include "pipes.h"
#include <ctype.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

///@todo error handling
tensor* pipe_tensor(char* command){
  FILE* pipe = popen(command, "r");
  tensor* tensor = read_tensor(pipe);
  pclose(pipe);
  return tensor;
}

tensor* pipe_kernel(param* p){
  char command[1024];

  double L = unitlength(p);
  sprintf(command, "kernel --size %d %d %d --msat %g --aexch %g --cellsize %g %g %g \n",
          p->size[X], p->size[Y], p->size[Z],
          p->msat, p->aexch,
          p->cellSize[X] * L, p->cellSize[Y] * L, p->cellSize[Z] * L);
          
  fprintf(stderr, "%s", command);
  return pipe_tensor(command);
}

#ifdef __cplusplus
}
#endif