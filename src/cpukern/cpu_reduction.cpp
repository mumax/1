#include "cpu_reduction.h"
#include <stdlib.h>
#include <math.h>
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

//begin cpu_sum ------------------------
typedef struct{
  float *input, *sum_t;
  int N;
} cpu_sum_arg;

void cpu_sum_t(int id){

  cpu_sum_arg *arg = (cpu_sum_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  arg->sum_t[id] = 0.0f;
  for(int i=start; i<stop; i++)
    arg->sum_t[id] += arg->input[i];

  return;
}

float cpu_sum(float* input, int N){

  float sum_t[T_data->N_threads];
  
  cpu_sum_arg args;
  args.input = input;
  args.N = N;
  args.sum_t = sum_t;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_sum_t);

  float sum = sum_t[0];
  for(int i=1; i<T_data->N_threads; i++){
    sum += sum_t[i];
  }
  
  return sum;
}
//end cpu_sum --------------------------

//begin cpu_max ------------------------
typedef struct{
  float *input, *max_t;
  int N;
} cpu_max_arg;

void cpu_max_t(int id){

  cpu_max_arg *arg = (cpu_max_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    if( arg->input[i] > arg->max_t[id])
      arg->max_t[id] = arg->input[i];

  return;
}

float cpu_max(float* input, int N){

  float max_t[T_data->N_threads];
  for (int i=0; i<T_data->N_threads; i++)
    max_t[i] = input[0];
  
  cpu_max_arg args;
  args.input = input;
  args.N = N;
  args.max_t = max_t;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_max_t);

  float max = max_t[0];
  for(int i=1; i<T_data->N_threads; i++){
    if( max_t[i] > max)
      max = max_t[i];
  }
  
  return max;
}
//end cpu_max --------------------------


//begin cpu_maxabs ---------------------
typedef struct{
  float *input, *maxabs_t;
  int N;
} cpu_maxabs_arg;

void cpu_maxabs_t(int id){

  cpu_maxabs_arg *arg = (cpu_maxabs_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  arg->maxabs_t[id] = 0.0f;
  for(int i=start; i<stop; i++)
    if( fabs(arg->input[i]) > arg->maxabs_t[id])
      arg->maxabs_t[id] = fabs(arg->input[i]);

  return;
}

float cpu_maxabs(float* input, int N){

  float maxabs_t[T_data->N_threads];
  
  cpu_maxabs_arg args;
  args.input = input;
  args.N = N;
  args.maxabs_t = maxabs_t;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_maxabs_t);

  float maxabs = maxabs_t[0];
  for(int i=1; i<T_data->N_threads; i++){
    if( maxabs_t[i] > maxabs)
      maxabs = maxabs_t[i];
  }
  
  return maxabs;
}
//end cpu_maxabs -----------------------


//begin cpu_min ------------------------
typedef struct{
  float *input, *min_t;
  int N;
} cpu_min_arg;

void cpu_min_t(int id){

  cpu_min_arg *arg = (cpu_min_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->N);

  for(int i=start; i<stop; i++)
    if( arg->input[i] < arg->min_t[id])
      arg->min_t[id] = arg->input[i];

  return;
}

float cpu_min(float* input, int N){

  float min_t[T_data->N_threads];
  for (int i=0; i<T_data->N_threads; i++)
    min_t[i] = input[0];
  
  cpu_min_arg args;
  args.input = input;
  args.N = N;
  args.min_t = min_t;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_min_t);

  float min = min_t[0];
  for(int i=1; i<T_data->N_threads; i++){
    if( min_t[i] < min)
      min = min_t[i];
  }
  
  return min;
}
//end cpu_min --------------------------




/// Reduces the input (array on device)
float cpu_reduce(int operation, float* input, float* devbuffer, float* hostbuffer, int blocks, int threadsPerBlock, int N){
  switch(operation){
    default: abort();
    case REDUCE_ADD: return cpu_sum(input, N);
    case REDUCE_MAX: return cpu_max(input, N);
    case REDUCE_MAXABS: return cpu_maxabs(input, N);
    case REDUCE_MIN: return cpu_min(input, N);
  }
}


#ifdef __cplusplus
}
#endif
