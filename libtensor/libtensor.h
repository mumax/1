/*
 * libtensor provides a common interface for N-dimensional arrays of floats. The tensor type knows its rank and sizes in each dimension, so these do not have to be passed to function calls. Tensors can also be input/output in a common format, for sharing them between different programs or storing them in files.
 */

#ifndef LIBTENSOR_H
#define LIBTENSOR_H

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>


typedef struct{
   int rank;			// tensor rank
   int* size; 			// array of length rank, stores dimensions in all directions 
   float* list;			// data as a continous array of length size[0]*size[1]*...
}tensor;


/** Creates a new tensor with given rank and size. Allocates the neccesary space for the elements. */
tensor* new_tensor(int rank, int* size);

tensor* new_tensor0();

/** An easy way to acces tensor elements is to view it as an N-dimensional array. */
// void* tensor_array(tensor* t){
// 
// }


/** Returns the address of element i,j,k,... inside the tensor. This can be used to set or get elements form the tensor. */
float* tensor_elem(tensor* t, int* index);

/** Given an N-dimensional index (i, j, k, ...), this function calculates the 1-dimensional index in the corresponding array that stores the tensor data. Thus, tensor_elem(i,j,k) is equivalent to list[tensor_index(i,j,k)]. */ 
int tensor_index(tensor* t, int* indexarray);

/** Returns the total number of elements in the tensor: size[0]*size[1]*...*size[rank-1]. This is also the length of the contiguous array that stores the tensor data. */
int tensor_length(tensor* t);

/** Frees everything. */
void delete_tensor(tensor* t);

/** Prints the tensor as ascii text */
void print_tensor(tensor* t, FILE* out);

tensor* read_tensor(FILE* in);

#ifdef __cplusplus
}
#endif

#endif