#ifndef LIBTENSOR_H
#define LIBTENSOR_H

#include <stdlib.h>
#include <stdio.h>

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif


/* Define SINGLE_PRECISSION or DOUBLE_PRECISSION. Can be overridden via a command-line flag. */
#ifndef SINGLE_PRECISSION
#ifndef DOUBLE_PRECISSION

#define SINGLE_PRECISSION

#endif
#endif


/** We will probably do everything with single-precission floating point data. Nevertheless we use 'real' instead of float so that later we might still re-define it as double. */
#ifdef SINGLE_PRECISSION
typedef float real;
#endif

#ifdef DOUBLE_PRECISSION
typedef double real;
#endif


typedef struct{
   int rank;			// tensor rank
   int* size; 			// array of length rank, stores dimensions in all directions 
   float* list;			// data as a continous array of length size[0]*size[1]*...
}tensor;


/** Creates a new tensor with given rank and size. Allocates the neccesary space for the elements. */
tensor* new_tensor(int rank, int* size);

/** Prints the tensor as ascii text */
void print_tensor(tensor* t);

#ifdef __cplusplus
}
#endif

#endif