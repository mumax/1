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
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif


//begin cpu_normalize_uniform --------
typedef struct{
  float *m;
  int N;
} cpu_normalize_uniform_arg;

void cpu_normalize_uniform_t(int id){

  cpu_normalize_uniform_arg *arg = (cpu_normalize_uniform_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  float* mx = &(arg->m[0*arg->N]);
  float* my = &(arg->m[1*arg->N]);
  float* mz = &(arg->m[2*arg->N]);

  for(int i=start; i<stop; i++){
    float norm = 1.0/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
  return;
}

void cpu_normalize_uniform(float* m, int N){

  cpu_normalize_uniform_arg args;
  args.m = m;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_normalize_uniform_t);

  return;
}
//end cpu_normalize_uniform ----------


//begin cpu_normalize_map ------------
typedef struct{
  float *m, *map;
  int N;
} cpu_normalize_map_arg;


void cpu_normalize_map_t(int id){

  cpu_normalize_map_arg *arg = (cpu_normalize_map_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  float* mx = &(arg->m[0*arg->N]);
  float* my = &(arg->m[1*arg->N]);
  float* mz = &(arg->m[2*arg->N]);

  for(int i=start; i<stop; i++){
    float norm = arg->map[i]/sqrt(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
    mx[i] *= norm;
    my[i] *= norm;
    mz[i] *= norm;
  }
  
  return;
}


void cpu_normalize_map(float* m, float* map, int N){
  
  cpu_normalize_map_arg args;
  args.m = m;
  args.map = map;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_normalize_map_t);

  return;
}
//end cpu_normalize_map --------------

#ifdef __cplusplus
}
#endif
