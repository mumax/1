/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_reduction.h"
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @todo parallellize
float cpu_sum(float* input, int N){
  float sum = 0.0;
  for(int i=0; i<N; i++){
    sum += input[i];
  }
  return sum;
}

/// @todo parallellize
float cpu_max(float* input, int N){
  float max = input[0];
  for(int i=1; i<N; i++){
    if( input[i] > max)
      max = input[i];
  }
  return max;
}

/// @todo parallellize
float cpu_maxabs(float* input, int N){
  float max = fabs(input[0]);
  for(int i=1; i<N; i++){
    if( fabs(input[i]) > max)
      max = fabs(input[i]);
  }
  return max;
}



/// @todo parallellize
float cpu_min(float* input, int N){
  float min = input[0];
  for(int i=1; i<N; i++){
    if( input[i] < min)
      min = input[i];
  }
  return min;
}

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
