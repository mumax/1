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
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif


//begin cpu_add ------------------------
typedef struct{
  float *a, *b;
  int N;
} cpu_add_arg;

void cpu_add_t(int id){

  cpu_add_arg *arg = (cpu_add_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    arg->a[i] += arg->b[i];

  //   saxpy(stop-start, 1.0f, arg->a + start, 1, arg->b + start, 1);

  return;
}

void cpu_add(float* a, float* b, int N){

  cpu_add_arg args;
  args.a = a;
  args.b = b;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_t);

  return;
}
//end cpu_add --------------------------


//begin cpu_madd -----------------------
typedef struct{
  float *a, *b;
  float cnst;
  int N;
} cpu_madd_arg;

void cpu_madd_t(int id){

  cpu_madd_arg *arg = (cpu_madd_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    arg->a[i] += arg->cnst*arg->b[i];
//   saxpy(stop-start, arg->cnst, arg->a + start, 1, arg->b + start, 1);
  
  return;
}

void cpu_madd(float* a, float cnst, float* b, int N){

  cpu_madd_arg args;
  args.a = a;
  args.cnst = cnst;
  args.b = b;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_madd_t);
  
  return;
}
//end cpu_madd -------------------------


//begin cpu_add_constant ---------------
typedef struct{
  float *a;
  float cnst;
  int N;
} cpu_add_constant_arg;

void cpu_add_constant_t(int id){

  cpu_add_constant_arg *arg = (cpu_add_constant_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    arg->a[i] += arg->cnst;
  
  return;
}

void cpu_add_constant(float* a, float cnst, int N){

  cpu_add_constant_arg args;
  args.a = a;
  args.cnst = cnst;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_add_constant_t);

  return;
}
//end cpu_add_constant -----------------


//begin cpu_linear_combination ---------
typedef struct{
  float *a, *b;
  float weightA, weightB;
  int N;
} cpu_linear_combination_arg;

void cpu_linear_combination_t(int id){

  cpu_linear_combination_arg *arg = (cpu_linear_combination_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    arg->a[i] = arg->weightA * arg->a[i] + arg->weightB * arg->b[i];

  return;
}

void cpu_linear_combination(float* a, float* b, float weightA, float weightB, int N){

  cpu_linear_combination_arg args;
  args.a = a;
  args.b = b;
  args.weightA = weightA;
  args.weightB = weightB;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_linear_combination_t);
  
  return;
}
//end cpu_linear_combination -----------


//begin cpu_scale_dot_product ----------
typedef struct{
  float *result, *vector1, *vector2;
  float a;
  int N;
} cpu_scale_dot_product_arg;

void cpu_scale_dot_product_t(int id){
  cpu_scale_dot_product_arg *arg = (cpu_scale_dot_product_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    arg->result[i] = arg->a*(arg->vector1[         i] * arg->vector2[         i] + 
                             arg->vector1[  arg->N+i] * arg->vector2[  arg->N+i] + 
                             arg->vector1[2*arg->N+i] * arg->vector2[2*arg->N+i]   );
  
  return;
}

void cpu_scale_dot_product(float* result, float *vector1, float *vector2, float a, int N){

  cpu_scale_dot_product_arg args;
  args.result = result;
  args.vector1 = vector1;
  args.vector2 = vector2;
  args.a = a;
  args.N = N;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_scale_dot_product_t);
  
  return;
}
//end cpu_scale_dot_product ------------



#ifdef __cplusplus
}
#endif
