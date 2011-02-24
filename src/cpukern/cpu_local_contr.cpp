/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

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
