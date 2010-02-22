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
#include <stdarg.h>


typedef struct{
   int32_t rank;		// tensor rank
   int32_t* size; 		// array of length rank, stores dimensions in all directions 
   float* list;			// data as a continous array of length size[0]*size[1]*...
}tensor;


/** Creates a new tensor with given rank and size (as integer vararg). Allocates the neccesary space for the elements. */
tensor* new_tensor(int rank, ...);

/** The same as new_tensor(), but with the size given as an array. This is handy when the rank is not know at compile time. */
tensor* new_tensorN(int rank, int* size);

/** Frees everything. */
void delete_tensor(tensor* t);


/** Returns the total number of elements in the tensor: size[0]*size[1]*...*size[rank-1]. This is also the length of the contiguous array that stores the tensor data. */
int tensor_length(tensor* t);


/** Returns the address of element i,j,k,... inside the tensor. This can be used to set or get elements form the tensor. */
float* tensor_elem(tensor* t, int* index);

/** Given an N-dimensional index (i, j, k, ...), this function calculates the 1-dimensional index in the corresponding array that stores the tensor data. Thus, tensor_elem(i,j,k) is equivalent to list[tensor_index(i,j,k)]. */ 
int tensor_index(tensor* t, int* indexarray);

/** An easy way to acces tensor's elements is to view it as an N-dimensional array. */
// void* tensor_array(tensor* t){
// 
// }


/** Writes the tensor in binary format */
void write_tensor(tensor* t, FILE* out);

/** Reads the tensor from binary format */
tensor* read_tensor(FILE* in);

/** Prints the tensor as ascii text. Can also be used to print to the screen: write_tensor_ascii(tensor, stdout); */
void write_tensor_ascii(tensor* t, FILE* out);

/** Prints the tensor to standard output. */
void print_tensor(tensor* t);


/** Makes a slice of a tensor, sharing the data with the original. */
tensor* tensor_component(tensor* t, int component);

/** Tensors created by tensor_component() should not be freed with delete_tensor() but with delete_tensor_component(), as the parent tensor may still be using the data list. */
void delete_tensor_component(tensor* t);

#ifdef __cplusplus
}
#endif

#endif