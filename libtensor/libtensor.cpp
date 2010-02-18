#include "libtensor.h"

#include <iostream>

using namespace std;

/** Creates a new tensor with given rank and size. Allocates the neccesary space for the elements. */
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

/** Prints the tensor as ascii text */
void print_tensor(tensor* t){
    cout << "#tensor" << endl;
}

