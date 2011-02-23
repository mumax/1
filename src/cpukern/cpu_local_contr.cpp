#include "cpu_local_contr.h"
#include "thread_functions.h"

#include "../macros.h"
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void cpu_add_local_fields_uniaxial(float* mx, float* my, float* mz,
                                              float* hx, float* hy, float* hz,
                                              float hext_x, float hext_y, float hext_z,
                                              float U0, float U1, float U2,
                                              int N){
}

typedef struct{
  float *hx, *hy, *hz;
  float hext_x, hext_y, hext_z;
  int N;
}cpu_add_external_field_arg;

void cpu_add_external_field_t(int id){
  
  cpu_add_external_field_arg *arg = (cpu_add_external_field_arg *) func_arg;

  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i = start; i < stop; i++){
    arg->hx[i] += arg->hext_x;
    arg->hy[i] += arg->hext_y;
    arg->hz[i] += arg->hext_z;
  }

  return;
}

void cpu_add_external_field(float* hx, float* hy, float* hz,
                            float hext_x, float hext_y, float hext_z,
                            int N){

  cpu_add_external_field_arg args;
  args.hx = hx;
  args.hy = hy;
  args.hz = hz;
  args.hext_x = hext_x;
  args.hext_y = hext_y;
  args.hext_z = hext_z;
  args.N = N;
  
  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_external_field_t);
  
  return;
}


void cpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){
	fprintf(stderr, "cpu_local_fields unimplemented\n");
	abort();
}


#ifdef __cplusplus
}
#endif
