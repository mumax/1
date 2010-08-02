#include "cpu_properties.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_print_properties(FILE* out){
//   int MiB = 1024 * 1024;
//   int kiB = 1024;
//   
  fprintf(out, "               CPU:");
  
}


void cpu_print_properties_stdout(){
  cpu_print_properties(stdout);
}

#ifdef __cplusplus
}
#endif
