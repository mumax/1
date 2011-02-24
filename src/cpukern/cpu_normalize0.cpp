/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_normalize.h"
#include <math.h>
#ifdef __cplusplus
extern "C" {
#endif


void cpu_normalize_uniform(float* m, int N){

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  #pragma omp parallel for
  for(int i=0; i<N; i++){
    float norm = 1.0/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
}


void cpu_normalize_map(float* m, float* map, int N){
  
  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  #pragma omp parallel for
  for(int i=0; i<N; i++){
    float norm = map[i]/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    if(map[i] == 0){  //HACK
      norm = 0.;
    }
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
}

#ifdef __cplusplus
}
#endif
