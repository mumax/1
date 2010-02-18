#include "libtensor.h"

#include <iostream>

using namespace std;


tensor* new_tensor(int rank, int* size){
  int i, totalsize;
  
  tensor* t = (tensor*)malloc(sizeof(tensor));
  
  t->rank = rank;
  
  t->size = (int*)calloc(rank, sizeof(int));	// we copy the size array to protect from accidental modification
						// also, if we're a bit lucky, it gets allocated nicely after t and before list,
						// so we can have good cache efficiency.
  totalsize = 1;
  for(i=0; i<rank; i++){
    t->size[i] = size[i];
    totalsize *= size[i];
  }
  
  t-> list = (float*)calloc(totalsize, sizeof(float));
  
  return t;
}


tensor* new_tensor0(){
  tensor* t = (tensor*)malloc(sizeof(tensor));
  t->rank = 0;
  t->size = NULL;
  t->list = (float*)calloc(1, sizeof(float));
}


int tensor_index(tensor* t, int* indexarray){
  int i;
  int index = indexarray[0];
  //AssertMsg(! (indexarray[0] < 0 || indexarray[0] >= size[0]), "Index out of range");
  for (i=1; i<t->rank; i++){
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
  int i;
  for(i=0; i < t->rank; i++){
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


/** Prints the tensor as ascii text */
void print_tensor(tensor* t){
    cout << "#tensor" << endl;
}

