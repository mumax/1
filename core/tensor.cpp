#include "tensor.h"
#include <iostream>
#include <assert.h>

using namespace std;


tensor* new_tensor(int rank, ...){
  
  tensor* t = (tensor*)safe_malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)safe_calloc(rank, sizeof(int32_t));	// we copy the size array to protect from accidental modification
							// also, if we're a bit lucky, it gets allocated nicely after t and before list,
							// so we can have good cache efficiency.
  va_list varargs;
  va_start(varargs, rank);
  
  for(int i=0; i<rank; i++){
    t->size[i] = va_arg(varargs, int32_t);
  }
  va_end(varargs);
  
  t-> list = (float*)safe_calloc(tensor_length(t), sizeof(float));
  
  return t;
}


tensor* as_tensor(float* list, int rank, ...){
  
  tensor* t = (tensor*)safe_malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)safe_calloc(rank, sizeof(int32_t));	
  
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
    cerr << "2nd argument (" << r << ")!= tensor rank (" << t->rank << ")" << endl;
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
  tensor* t = (tensor*)safe_malloc(sizeof(tensor));
  t->rank = rank;
  t->size = (int*)safe_calloc(rank, sizeof(int32_t));
  
  for(int i=0; i<rank; i++){
    t->size[i] = size[i];
  }
 
  t-> list = (float*)safe_calloc(tensor_length(t), sizeof(float));
  
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
  float** sliced = (float**)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = &list[i*size1];
  }
  return sliced;
}

float*** slice_array3D(float* list, int size0, int size1, int size2){
  float*** sliced = (float***)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = (float**)safe_calloc(size1, sizeof(float*));
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      sliced[i][j] = &list[ (i * size1 + j) *size2 + 0];
    }
  }
  return sliced;
}

float*** tensor_array3D(tensor* t){
  assert(t->rank == 3);
  return slice_array3D(t->list, t->size[0], t->size[1], t->size[2]);
}

float**** slice_array4D(float* list, int size0, int size1, int size2, int size3){
  float**** sliced = (float****)safe_calloc(size0, sizeof(float*));
  for(int i=0; i < size0; i++){
    sliced[i] = (float***)safe_calloc(size1, sizeof(float*));
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      sliced[i][j] = (float**)safe_calloc(size2, sizeof(float*));
    }
  }
  for(int i=0; i<size0; i++){
    for(int j=0; j<size1; j++){
      for(int k=0; k<size2; k++){
	sliced[i][j][k] = &list[ ((i * size1 + j) *size2 + k) * size3 + 0];
      }
    }
  }
  return sliced;
}

float**** tensor_array4D(tensor* t){
  assert(t->rank == 4);
  return slice_array4D(t->list, t->size[0], t->size[1], t->size[2], t->size[3]);
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
  write_int(rank, out);
  for(int i=0; i<rank; i++){
    write_int(size[i], out);
  }
  fwrite(list, sizeof(float), length, out);
  // todo: error handling
}


void write_int(int i, FILE* out){
  int32_t i32 = (int32_t)i;
  fwrite(&i32, sizeof(int32_t), 1, out);
}

int read_int(FILE* in){
  int32_t i;
  fread(&i, sizeof(int32_t), 1, in);
  return (int)i;
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
  // print rank...
  fprintf(out, "%d\n", t->rank);
  // ... and size...
  for(int i=0; i < t->rank; i++){
    fprintf(out, "%d\n", t->size[i]);
  }
  // ... and data
  for(int i=0; i < tensor_length(t); i++){
    fprintf(out, "% 11f ", t->list[i]);
    // If we reach the end of dimension, we print an extra newline
    // for each dimension:
    for(int j=0; j < t->rank; j++){
      // calc. the length in that dimension
      int size = 1;
      for(int k=j; k < t->rank; k++){
	size *= t->size[k];
      }
      // if we are at the end of the dimesion, print the newline.
      if((i+1) % size == 0){
	fprintf(out, "\n");
      }
    }
  }
}


void print_tensor(tensor* t){
  write_tensor_ascii(t, stdout);
}

// TODO: catch file not found!
tensor* read_tensor(FILE* in){
  tensor* t = (tensor*)safe_malloc(sizeof(tensor));
  read_tensor_pieces(&(t->rank), &(t->size), &(t->list), in);
  return t;
}

tensor* read_tensor_fname(char* filename){
  FILE* file = fopen(filename, "rb");
  if(file == NULL){
    fprintf(stderr, "Could not read file: %s", filename);
    abort();
  }
  return read_tensor(file);
  fclose(file);
}


void read_tensor_pieces(int* rank, int** size, float** list, FILE* in){
  //fread(rank, sizeof(int32_t), 1, in);
  (*rank) = read_int(in);
  (*size) = (int*)safe_calloc(*rank, sizeof(int));
  for(int i=0; i<(*rank); i++){
    (*size)[i] = read_int(in);
  }
  
  
  int length = 1;
  for(int i=0; i<*rank; i++){
    length *= (*size)[i];
  } 
  *list = (float*)safe_calloc(length, sizeof(float));
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


void* safe_malloc(int size){
  void* ptr = malloc(size);
  if(ptr == NULL){
    fprintf(stderr, "could not malloc(%d)\n", size);
    float trigger_segfault = *((float*)0);
  }
  return ptr;
}

void* safe_calloc(int length, int elemsize){
  void* ptr = calloc(length, elemsize);
  if(ptr == NULL){
    fprintf(stderr, "could not calloc(%d, %d)\n", length, elemsize);
    float trigger_segfault = *((float*)0);
  }
  return ptr;
}
