#include "tensor.h"
#include <iostream>
#include <assert.h>

using namespace std;


tensor* new_tensor(int rank, ...){
  
  tensor* t = (tensor*)malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)calloc(rank, sizeof(int32_t));	// we copy the size array to protect from accidental modification
							// also, if we're a bit lucky, it gets allocated nicely after t and before list,
							// so we can have good cache efficiency.
  va_list varargs;
  va_start(varargs, rank);
  
  for(int i=0; i<rank; i++){
    t->size[i] = va_arg(varargs, int32_t);
  }
  va_end(varargs);
  
  t-> list = (float*)calloc(tensor_length(t), sizeof(float));
  
  return t;
}


tensor* as_tensor(float* list, int rank, ...){
  
  tensor* t = (tensor*)malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)calloc(rank, sizeof(int32_t));	
  
  va_list varargs;
  va_start(varargs, rank);
  
  for(int i=0; i<rank; i++){
    t->size[i] = va_arg(varargs, int32_t);
  }
  va_end(varargs);
  
  t-> list = list;
  
  return t;
}


float* tensor_get(tensor* t, int r ...){
  int* index = new int[t->rank];
  
  va_list varargs;
  va_start(varargs, r);
  if(r != t->rank){
    cerr << "2nd argument != tensor rank" << endl;
    exit(-3);
  }
  
  for(int i=0; i<t->rank; i++){
    index[i] = va_arg(varargs, int);
  }
  va_end(varargs);
  float* ret = tensor_elem(t, index);
  delete[] index;
  return ret;
}


tensor* new_tensorN(int rank, int* size){
  tensor* t = (tensor*)malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)calloc(rank, sizeof(int32_t));
  
  for(int i=0; i<rank; i++){
    t->size[i] = size[i];
  }
 
  t-> list = (float*)calloc(tensor_length(t), sizeof(float));
  
  return t;
}


int tensor_index(tensor* t, int* indexarray){
  int index = indexarray[0];
  assert(! (indexarray[0] < 0 || indexarray[0] >= t->size[0]));
  for (int i=1; i<t->rank; i++){
    assert(!(indexarray[i] < 0 || indexarray[i] >= t->size[i]));
    index *= t->size[i];
    index += indexarray[i];
  }
  return index;
}


float* tensor_elem(tensor* t, int* indexarray){
  return &(t->list[tensor_index(t, indexarray)]);
}


float** tensor_array2D(tensor* t){
  assert(t->rank == 2);
  return slice_array2D(t->list, t->size[0], t->size[1]);
}

float** slice_array2D(float* list, int size0, int size1){
  float** sliced = (float**)calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = &list[i*size1];
  }
  return sliced;
}


int tensor_length(tensor* t){
  int length = 1;
  for(int i=0; i < t->rank; i++){
    length *= t->size[i]; 
  }
  return length;
}


void delete_tensor(tensor* t){
  // for safety, we invalidate the tensor so we'd quickly notice accidental use after freeing.
  t->rank = -1;
  t->size = NULL;
  t->list = NULL;
  free(t->size);
  free(t->list);
  free(t);
}


void write_tensor(tensor* t, FILE* out){
  write_tensor_pieces(t->rank, t->size, t->list, out);
}


void write_tensor_pieces(int rank, int* size, float* list, FILE* out){
  int length = 1;
  for(int i=0; i<rank; i++){
    length *= size[i];
  }  
  fwrite(&(rank), sizeof(int32_t), 1, out);
  fwrite(size, sizeof(int32_t), rank, out);
  fwrite(list, sizeof(float), length, out);
  // todo: error handling
}


void write_int(int i, FILE* out){
  fwrite(&i, sizeof(int32_t), 1, out);
}


void write_float(float f, FILE* out){
  fwrite(&f, sizeof(int32_t), 1, out);
}


void write_tensor_ascii(tensor* t, FILE* out){
  fprintf(out, "%d\n", t->rank);
  for(int i=0; i<t->rank; i++){
    fprintf(out, "%d\n", t->size[i]);
  }
  for(int i=0; i<tensor_length(t); i++){
    fprintf(out, "%f\n", t->list[i]);
  }
}


void format_tensor(tensor* t, FILE* out){
  if(t->rank != 3){
    write_tensor_ascii(t, out);
    return;
  }
  
  for(int i=0; i<t->size[0]; i++){
    for(int j=0; j<t->size[1]; j++){
      for(int k=0; k<t->size[2]; k++){
	fprintf(out, "%f ", *tensor_get(t, 3, i, j, k));
      }
      fprintf(out, "\n");
    }
    fprintf(out, "\n");
  }
  
}


void print_tensor(tensor* t){
  write_tensor_ascii(t, stdout);
}


tensor* read_tensor(FILE* in){
  tensor* t = (tensor*)malloc(sizeof(tensor));
  read_tensor_pieces(&(t->rank), &(t->size), &(t->list), in);
  return t;
}


void read_tensor_pieces(int* rank, int** size, float** list, FILE* in){
  fread(rank, sizeof(int32_t), 1, in);
  *size = (int32_t*)calloc(*rank, sizeof(int32_t));
  fread(*size, sizeof(int32_t), *rank, in);
  int length = 1;
  for(int i=0; i<*rank; i++){
    length *= (*size)[i];
  } 
  *list = (float*)calloc(length, sizeof(float));
  fread(*list, sizeof(float), length, in);
}


tensor* tensor_component(tensor* t, int component){
  int* size = new int[t->rank-1];
  for(int i=0; i<t->rank-1; i++){
    size[i] = t->size[i+1];
  }
  tensor* slice = new_tensorN(t->rank-1, size);
  delete[] size;
  int* index = new int[t->rank];
  for(int i=1; i<t->rank; i++){
    index[i] = 0;
  }
  index[0] = component;
  slice->list = tensor_elem(t, index);
  delete[] index;
  return slice;
}


void delete_tensor_component(tensor* t){
  // for safety, we invalidate the tensor so we'd quickly notice accidental use after freeing.
  t->rank = -1;
  t->size = NULL;
  t->list = NULL;
  free(t->size);
  // we do not free t->list as it is owned by the parent tensor who may still be using it.
  free(t);
}
