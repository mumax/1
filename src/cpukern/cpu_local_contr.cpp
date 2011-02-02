#include "cpu_local_contr.h"
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


void cpu_add_external_field(float* hx, float* hy, float* hz,
                                        float hext_x, float hext_y, float hext_z,
                                        int N){
  #pragma omp parallel for
  for(int i = 0; i < N; i++){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}


void cpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){
	fprintf(stderr, "cpu_local_fields unimplemented\n");
	abort();
}


#ifdef __cplusplus
}
#endif
