#include "libtensor.h"


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


int tensor_index(tensor* t, int* indexarray){
  int index = indexarray[0];
  //AssertMsg(! (indexarray[0] < 0 || indexarray[0] >= size[0]), "Index out of range");
  for (int i=1; i<t->rank; i++){
    //AssertMsg(!(indexarray[i] < 0 || indexarray[i] >= size[i]), "Index out of range");
    index *= t->size[i];
    index += indexarray[i];
  }
  return index;
}


float* tensor_elem(tensor* t, int* indexarray){
  return &(t->list[tensor_index(t, indexarray)]);
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
  fwrite(&(t->rank), sizeof(int32_t), 1, out);
  fwrite(t->size, sizeof(int32_t), t->rank, out);
  fwrite(t->list, sizeof(float), tensor_length(t), out);
  // todo: error handling
}

tensor* read_tensor(FILE* in){
  tensor* t = (tensor*)malloc(sizeof(tensor));
  fread(&(t->rank), sizeof(int32_t), 1, in);
  t->size = (int32_t*)calloc(t->rank, sizeof(int32_t));
  fread(t->size, sizeof(int32_t), t->rank, in);
  t-> list = (float*)calloc(tensor_length(t), sizeof(float));
  fread(t->list, sizeof(float), tensor_length(t), in);
  return t;
}