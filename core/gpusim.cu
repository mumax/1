#include "gpusim.h"
#include <stdio.h>
#include <assert.h>

void gpusim_checksize_m(gpusim* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpusim_loadm(gpusim* sim, tensor* m){
  gpusim_checksize_m(sim, m); 
  memcpy_to_gpu(m->list, sim->m, sim->len_m);
}

void gpusim_dumpm(gpusim* sim, tensor* m){
   gpusim_checksize_m(sim, m);
   memcpy_from_gpu(sim->m, m->list, sim->len_m);
}


gpusim* new_gpusim(int N0, int N1, int N2){
  gpusim* sim = (gpusim*)malloc(sizeof(gpusim));
  
  sim->size = (int*)calloc(3, sizeof(int));
  sim->size[0] = N0; sim->size[1] = N1; sim->size[2] = N2;
  sim->N = N0 * N1 * N2;
  
  sim->len_m = 3 * sim->N;
  sim->m = new_cuda_array(sim->len_m);
  
  return sim;
}


void memcpy_to_gpu(float* source, float* dest, int nElements){
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyHostToDevice);
  if(status != 0){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to device addres %p", nElements, source, dest);
  }
}


void memcpy_from_gpu(float* source, float* dest, int nElements){
  int status = cudaMemcpy(dest, source, nElements*sizeof(float), cudaMemcpyDeviceToHost);
  if(status != 0){
    fprintf(stderr, "CUDA could not copy %d floats from host addres %p to device addres %p", nElements, source, dest);
  }
}


float* new_cuda_array(int size){
  float* array = NULL;
  int status = cudaMalloc((void**)(&array), size * sizeof(float));
  if(status != 0 || array == NULL){
    fprintf(stderr, "CUDA could not allocate %d bytes\n", size);
    abort();
  }
  return array;
}
