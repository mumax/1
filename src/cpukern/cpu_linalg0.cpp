/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_linalg.h"

#ifdef __cplusplus
extern "C" {
#endif


void cpu_add(float* a, float* b, int N){
  #pragma omp parallel for
  for(int i=0; i<N; i++){
    a[i] += b[i];
  }
}

void cpu_madd(float* a, float cnst, float* b, int N){
  #pragma omp parallel for
  for(int i=0; i<N; i++){
    a[i] += cnst * b[i];
  }
}


void cpu_add_constant(float* a, float cnst, int N){
  #pragma omp parallel for
  for(int i=0; i<N; i++){
    a[i] += cnst;
  }
}


void cpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){
  #pragma omp parallel for
  for(int i=0; i<N; i++){
    a[i] = weightA * a[i] + weightB * b[i];
  }
}

#ifdef __cplusplus
}
#endif
