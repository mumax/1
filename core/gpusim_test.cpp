#include "gpusim.h"
#include "tensor.h"

#include <stdio.h>

int main(int argc, char** argv){
  printf("gpusim_test\n");
  
  int N0 = 8, N1 = 8, N2 = 4;
  gpusim* sim = new_gpusim(N0, N1, N2);
  
  tensor* m = new_tensor(4, 3, N0, N1, N2);
  format_tensor(m, stdout);
  
  gpusim_loadm(sim, m);
  gpusim_dumpm(sim, m);
  
  format_tensor(m, stdout);
  
  return 0;
}