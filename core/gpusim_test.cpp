#include "gpusim.h"
#include "tensor.h"

#include <stdio.h>

int main(int argc, char** argv){
  printf("gpusim_test\n");
  
  FILE* mfile = fopen(argv[1], "rb");
  tensor* m = read_tensor(mfile);
  fclose(mfile);
  int N0 = m->size[1];
  int N1 = m->size[2];
  int N2 = m->size[3];
  printf("read m: %d x %d x %d\n", N0, N1, N2);
  
  FILE* kernelfile = fopen(argv[2], "rb");
  tensor* kernel = read_tensor(kernelfile);
  fclose(kernelfile);
  printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
  
  gpusim* sim = new_gpusim(N0, N1, N2, kernel);
  for(int i=0; i<sim->len_m; i++){ m->list[i] = 1.; }
  
  gpusim_loadm(sim, m);
  gpusim_updateh(sim);
  
  tensor* ft_m_i = new_tensor(3, 2*N0, 2*N1, 2*2*N2);
  memcpy_from_gpu(sim->ft_m_i, ft_m_i->list, tensor_length(ft_m_i));
  
  format_tensor(ft_m_i, stdout);
  //gpusim_storem(sim, m);
  //format_tensor(m, stdout);
  
  
  return 0;
}