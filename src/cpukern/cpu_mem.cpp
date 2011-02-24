/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "cpu_mem.h"
#include "../macros.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

///@todo use fftwf_malloc
float* new_cpu_array(int size){
  assert(size > 0);
  float* array = (float*)malloc(size * sizeof(float));
  if(array == NULL){
    fprintf(stderr, "could not allocate %d floats in main memory\n", size);
    abort();
  }
  return array;
}

///@todo use fftwf_free
void free_cpu_array(float* array){
  free(array);
}

void cpu_zero(float* data, int nElements){

  memset(data, 0, nElements*sizeof(float));
  return;
}

void cpu_memcpy(float* source, float* dest, int nElements){

  memcpy(dest, source, nElements*sizeof(float));
  return;
}


float cpu_array_get(float* dataptr, int index){
  return dataptr[index];
}


void cpu_array_set(float* dataptr, int index, float value){
  dataptr[index] = value;
}

///@internal
int _cpu_stride_float_cache = -1;

/// We use quadword alignment by default, but allow to override just like on the GPU
int cpu_stride_float(){
  if( _cpu_stride_float_cache == -1){
    _cpu_stride_float_cache = 1; /// The default for now. @todo: this should become 4 when strided FFT's work
  }
  return _cpu_stride_float_cache;
}


void cpu_override_stride(int nFloats){
  assert(nFloats > -2);
  debugv( fprintf(stderr, "CPU stride overridden to %d floats\n", nFloats) );
  _cpu_stride_float_cache = nFloats;
}


#ifdef __cplusplus
}
#endif
