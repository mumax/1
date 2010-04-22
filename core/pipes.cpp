#include "pipes.h"
#include <ctype.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

tensor* pipe_tensor(char* command){
  FILE* pipe = popen(command, "r");
  tensor* tensor = read_tensor(pipe);
  pclose(pipe);
  return tensor;
}

#ifdef __cplusplus
}
#endif